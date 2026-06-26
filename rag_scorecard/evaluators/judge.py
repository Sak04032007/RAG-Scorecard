import os
from typing import Optional
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
try:
    import anthropic
except ImportError:
    anthropic = None
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

from rag_scorecard.models import RAGSample, MetricResult
from rag_scorecard.evaluators.tracker import LatencyCostTracker

class EvaluationVerdict(BaseModel):
    score: float = Field(..., description="A float metric score strictly graded between 0.0 and 1.0.")
    passed: bool = Field(..., description="True if performance meets acceptable enterprise baselines, False otherwise.")
    reasoning: str = Field(..., description="Surgical structural breakdown justifying the analytical verdict score.")

class MultiProviderEvaluatorJudge:
    def __init__(self, provider: str = "openai", model: Optional[str] = None, tracker: Optional[LatencyCostTracker] = None):
        self.provider = provider.lower()
        self.tracker = tracker

        if self.provider == "openai":
            if not OpenAI: raise ImportError("OpenAI SDK not installed.")
            self.model = model or "gpt-4o-mini"
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key: raise ValueError("OPENAI_API_KEY env var not set.")
            self.client = OpenAI(api_key=api_key)

        elif self.provider == "anthropic":
            if not anthropic: raise ImportError("Anthropic SDK not installed.")
            self.model = model or "claude-3-5-haiku-20241022"
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key: raise ValueError("ANTHROPIC_API_KEY env var not set.")
            self.client = anthropic.Anthropic(api_key=api_key)

        elif self.provider == "gemini":
            if not genai: raise ImportError("Google GenAI SDK not installed.")
            self.model = model or "gemini-1.5-flash"
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key: raise ValueError("GEMINI_API_KEY env var not set.")
            self.client = genai.Client(api_key=api_key)
        else:
            raise ValueError(f"Unrecognized provider specified: {provider}")

    def evaluate_context_relevance(self, sample: RAGSample) -> MetricResult:
        system_prompt = (
            "You are an expert infrastructure auditing judge grading information retrieval quality.\n"
            "Analyze the given User Query and the corresponding Retrieved Contexts chunks.\n"
            "Determine if the retrieved text contains direct support needed to answer the query.\n"
            "Output strictly formatted as a structured JSON object."
        )
        user_content = f"User Query: {sample.query}\n\nRetrieved Contexts:\n" + "\n---\n".join(sample.retrieved_contexts)
        return self._execute_llm_call("Context Relevance", system_prompt, user_content)

    def evaluate_faithfulness(self, sample: RAGSample) -> MetricResult:
        if not sample.generated_response:
            return MetricResult(metric_name="Faithfulness", score=0.0, metadata={"error": "No generated response string detected."})
        system_prompt = (
            "You are an AI pipeline reliability auditor grading RAG hallucination parameters.\n"
            "Compare the Generated Response string against the provided baseline Retrieved Contexts.\n"
            "If any fact, claim, or assertion in the answer is missing from the reference contexts, mark it down severely.\n"
            "Output strictly formatted as a structured JSON object."
        )
        user_content = f"Retrieved Contexts:\n" + "\n---\n".join(sample.retrieved_contexts) + f"\n\nGenerated Response: {sample.generated_response}"
        return self._execute_llm_call("Faithfulness", system_prompt, user_content)

    def evaluate_answer_relevance(self, sample: RAGSample) -> MetricResult:
        if not sample.generated_response:
            return MetricResult(metric_name="Answer Relevance", score=0.0, metadata={"error": "No generated response string detected."})
        system_prompt = (
            "You are an evaluation judge assessing RAG response focus and applicability.\n"
            "Compare the Generated Response directly against the User Query.\n"
            "Determine if the answer completely and directly addresses the core user request, penalizing redundant or non-sequitur fluff.\n"
            "Output strictly formatted as a structured JSON object."
        )
        user_content = f"User Query: {sample.query}\n\nGenerated Response: {sample.generated_response}"
        return self._execute_llm_call("Answer Relevance", system_prompt, user_content)

    def evaluate_completeness(self, sample: RAGSample) -> MetricResult:
        if not sample.generated_response or not sample.ground_truth:
            return MetricResult(metric_name="Completeness", score=0.0, metadata={"error": "Missing generated response or ground truths parameters."})
        system_prompt = (
            "You are a validation auditor measuring text accuracy parameters.\n"
            "Compare the Generated Response against the target canonical Ground Truth reference solutions.\n"
            "Grade whether all core factual concepts, constraints, or design requirements mentioned in the ground truths are successfully addressed.\n"
            "Output strictly formatted as a structured JSON object."
        )
        user_content = f"Ground Truth References:\n" + sample.ground_truth + f"\n\nGenerated Response: {sample.generated_response}"
        return self._execute_llm_call("Completeness", system_prompt, user_content)

    def _execute_llm_call(self, metric_name: str, system_prompt: str, user_content: str) -> MetricResult:
        try:
            if self.provider == "openai":
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
                    response_format=EvaluationVerdict,
                    temperature=0.0
                )
                verdict = response.choices[0].message.parsed
                if self.tracker and response.usage:
                    self.tracker.add_tokens(response.usage.prompt_tokens, response.usage.completion_tokens)

            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    temperature=0.0,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_content}],
                    tools=[{
                        "name": "submit_evaluation_verdict",
                        "description": "Submit data schema for verification grading.",
                        "input_schema": EvaluationVerdict.model_json_schema()
                    }],
                    tool_choice={"type": "tool", "name": "submit_evaluation_verdict"}
                )
                tool_use = [block for block in response.content if block.type == "tool_use"][0]
                verdict = EvaluationVerdict(**tool_use.input)
                if self.tracker and response.usage:
                    self.tracker.add_tokens(response.usage.input_tokens, response.usage.output_tokens)

            elif self.provider == "gemini":
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=EvaluationVerdict,
                )
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=user_content,
                    config=config
                )
                verdict = EvaluationVerdict.model_validate_json(response.text)
                if self.tracker and response.usage_metadata:
                    self.tracker.add_tokens(response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count)

            return MetricResult(metric_name=metric_name, score=verdict.score, metadata={"passed": verdict.passed, "reasoning": verdict.reasoning, "model_utilized": self.model})
        except Exception as e:
            return MetricResult(metric_name=metric_name, score=0.0, metadata={"error": f"LLM Judge API Engine failure execution: {str(e)}"})
