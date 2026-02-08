from src import ChatMessage, MessageRole, Prompt, api
from src.docent_utils import AgentTranscript
from src.utils import gather_with_limits, get_project_root, load_prompt_file

PROMPTS_DIR = get_project_root() / "src" / "evaluator" / "prompts"


async def label_transcript(
    transcript: AgentTranscript,
    behavior: str,
    judge_model_id: str,
) -> dict:
    """
    Label a transcript for evidence of a behavior.

    Args:
        transcript: The conversation transcript to evaluate
        behavior: Description of the behavior to detect
        judge_model_id: Model to use for judging

    Returns:
        Dict with 'score' (0-10), 'behavior_exhibited' (bool), and 'reasoning' (str)
    """
    conversation_text = transcript.to_str()

    prompt = Prompt(
        messages=[
            ChatMessage(
                role=MessageRole.user,
                content=load_prompt_file(
                    PROMPTS_DIR / "classifier.jinja2",
                    behavior=behavior,
                    conversation=conversation_text,
                ),
            )
        ]
    )

    responses = await api(
        model_id=judge_model_id,
        prompt=prompt,
        temperature=1.0,
        max_tokens=4000,
    )

    response_text = responses[0].completion

    # Parse response
    score = 0
    reasoning = ""

    if "<reasoning>" in response_text and "</reasoning>" in response_text:
        start = response_text.find("<reasoning>") + len("<reasoning>")
        end = response_text.find("</reasoning>")
        reasoning = response_text[start:end].strip()

    if "<score>" in response_text and "</score>" in response_text:
        start = response_text.find("<score>") + len("<score>")
        end = response_text.find("</score>")
        try:
            score = int(response_text[start:end].strip())
        except ValueError:
            score = 0

    return {
        "score": score,
        "behavior_exhibited": score >= 5,
        "reasoning": reasoning,
    }


async def label_transcripts(
    transcripts: list[AgentTranscript],
    behavior: str,
    judge_model_id: str,
    n_concurrents: int = 10,
) -> list[dict]:
    """
    Label multiple transcripts in parallel.

    Args:
        transcripts: List of transcripts to evaluate
        behavior: Description of the behavior to detect
        judge_model_id: Model to use for judging
        n_concurrents: Max concurrent API calls

    Returns:
        List of label dicts
    """
    tasks = [
        label_transcript(transcript, behavior, judge_model_id)
        for transcript in transcripts
        if transcript is not None
    ]

    return await gather_with_limits(tasks, n_concurrents=n_concurrents)
