from pathlib import Path

from src import ChatMessage, MessageRole, Prompt, api
from src.utils import get_project_root, load_prompt_file


class BasicModel:
    """A basic model wrapper around the inference API."""

    def __init__(
        self,
        name: str,
        model_id: str,
        temperature: float = 1.0,
        max_tokens: int = 2000,
    ):
        """
        Initialize the model.

        Args:
            name: Name of the model organism
            model_id: Model identifier to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens for generation
        """
        self.name = name
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def __call__(self, prompt: Prompt) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: The prompt to send to the model

        Returns:
            The model's response text
        """
        responses = await api(
            model_id=self.model_id,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return responses[0].completion


class SystemPromptModel(BasicModel):
    """A model with a custom system prompt injected."""

    def __init__(
        self,
        name: str,
        model_id: str,
        system_prompt: str | None = None,
        system_prompt_path: str | Path | None = None,
        temperature: float = 1.0,
        max_tokens: int = 2000,
    ):
        """
        Initialize the model with a system prompt.

        Args:
            name: Name of the model organism
            model_id: Model identifier to use
            system_prompt: Direct system prompt string
            system_prompt_path: Path to a file containing the system prompt
            temperature: Temperature for generation
            max_tokens: Maximum tokens for generation
        """
        super().__init__(
            name=name, model_id=model_id, temperature=temperature, max_tokens=max_tokens
        )

        if system_prompt and system_prompt_path:
            raise ValueError("Cannot specify both system_prompt and system_prompt_path")

        if system_prompt_path:
            self.system_prompt = load_prompt_file(system_prompt_path).strip()
        elif system_prompt:
            self.system_prompt = system_prompt.strip()
        else:
            raise ValueError("Must specify either system_prompt or system_prompt_path")

    @property
    def hidden_behavior(self) -> str:
        """Extract the hidden behavior from the system prompt's <hidden_behavior> tags."""
        if (
            "<hidden_behavior>" in self.system_prompt
            and "</hidden_behavior>" in self.system_prompt
        ):
            start = self.system_prompt.find("<hidden_behavior>") + len(
                "<hidden_behavior>"
            )
            end = self.system_prompt.find("</hidden_behavior>")
            return self.system_prompt[start:end].strip()
        return ""

    def _prepare_prompt(self, prompt: Prompt) -> Prompt:
        """Inject system prompt into the conversation."""
        messages = []
        original_messages = prompt.messages.copy()

        if original_messages and original_messages[0].role == MessageRole.system:
            # Prepend to existing system message
            combined = self.system_prompt + "\n\n" + original_messages[0].content
            messages.append(ChatMessage(role=MessageRole.system, content=combined))
            messages.extend(original_messages[1:])
        else:
            # Add system prompt as first message
            messages.append(
                ChatMessage(role=MessageRole.system, content=self.system_prompt)
            )
            messages.extend(original_messages)

        return Prompt(messages=messages)

    async def __call__(self, prompt: Prompt) -> str:
        """
        Generate a response with the system prompt injected.

        Args:
            prompt: The prompt to send to the model

        Returns:
            The model's response text
        """
        prepared_prompt = self._prepare_prompt(prompt)
        responses = await api(
            model_id=self.model_id,
            prompt=prepared_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return responses[0].completion
