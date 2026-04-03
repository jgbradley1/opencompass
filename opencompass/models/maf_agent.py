import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Union

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module()
class MAFAgent(BaseAPIModel):
    """Model wrapper around a Microsoft Agent Framework agent.

    This wrapper bridges OpenCompass evaluations with agents built using the
    Microsoft Agent Framework (``agent-framework`` package).  It supports
    multiple LLM providers exposed through MAF's client abstraction, including
    OpenAI, Azure OpenAI (chat and responses), and Azure AI Foundry.

    Args:
        path (str): The model / deployment name passed to the MAF client
            (e.g. ``'gpt-4o-mini'``).
        client_type (str): Which MAF client class to instantiate.  One of
            ``'openai'``, ``'azure_openai'``, ``'azure_responses'``, or
            ``'azure_ai'``.  Defaults to ``'openai'``.
        agent_name (str): Display name for the MAF agent. Defaults to
            ``'OpenCompassAgent'``.
        agent_instructions (str): System-level instructions forwarded to the
            MAF agent.  Defaults to an empty string (no extra instructions).
        api_key (str): API key.  When set to ``'ENV'`` the key is read from
            the relevant environment variable (``OPENAI_API_KEY`` or
            ``AZURE_OPENAI_API_KEY``).  Defaults to ``'ENV'``.
        endpoint (str, optional): Azure OpenAI / Azure AI endpoint URL.
            Required for Azure client types.
        project_endpoint (str, optional): Azure AI Foundry project endpoint
            URL (e.g. from Azure AI Studio).  Used by the ``'azure_ai'``
            client type as an alternative to ``endpoint``.
        deployment_name (str, optional): Azure deployment name.  Defaults to
            ``path`` when not provided.
        api_version (str, optional): Azure API version string.
        credential (object, optional): An Azure ``TokenCredential`` instance
            (e.g. ``DefaultAzureCredential``, ``AzureCliCredential``) used
            for Azure client types.  When *no* API key is found and no
            explicit credential is passed, the wrapper automatically falls
            back to ``DefaultAzureCredential`` from the ``azure-identity``
            package.
        tools (list, optional): A list of Python callables to register as
            tools on the MAF agent.  Each callable should follow MAF's tool
            conventions (typed parameters with ``Annotated`` / ``Field``).
        temperature (float, optional): Sampling temperature override.
        max_out_len (int): Default maximum output tokens.  Defaults to 512.
        query_per_second (int): Rate-limit for API calls.  Defaults to 1.
        retry (int): Number of retries on transient failures.  Defaults to 2.
        max_seq_len (int): Maximum sequence length budget.  Defaults to 4096.
        meta_template (Dict, optional): OpenCompass meta prompt template.
        max_workers (int, optional): Thread-pool size for batch generation.
            Defaults to ``min(32, (cpu_count + 5) * 2)``.
        verbose (bool): Whether to emit debug-level logs.  Defaults to False.
    """

    is_api: bool = True

    def __init__(
        self,
        path: str = 'gpt-4o-mini',
        client_type: str = 'openai',
        agent_name: str = 'OpenCompassAgent',
        agent_instructions: str = '',
        api_key: str = 'ENV',
        endpoint: Optional[str] = None,
        project_endpoint: Optional[str] = None,
        deployment_name: Optional[str] = None,
        api_version: Optional[str] = None,
        credential: Optional[object] = None,
        tools: Optional[List[Callable]] = None,
        temperature: Optional[float] = None,
        max_out_len: int = 512,
        query_per_second: int = 1,
        retry: int = 2,
        max_seq_len: int = 4096,
        meta_template: Optional[Dict] = None,
        max_workers: Optional[int] = None,
        verbose: bool = False,
    ):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            query_per_second=query_per_second,
            meta_template=meta_template,
            retry=retry,
            verbose=verbose,
        )

        self.client_type = client_type
        self.agent_name = agent_name
        self.agent_instructions = agent_instructions
        self.api_key = api_key
        self.endpoint = endpoint
        self.project_endpoint = project_endpoint
        self.deployment_name = deployment_name or path
        self.api_version = api_version
        self.credential = credential
        self.tools = tools or []
        self.temperature = temperature
        self.default_max_out_len = max_out_len

        if max_workers is None:
            import os
            cpu_count = os.cpu_count() or 1
            self.max_workers = min(32, (cpu_count + 5) * 2)
        else:
            self.max_workers = max_workers

        # Lazily built on first generate() call.
        self._agent = None
        # Persistent event loop shared across all _generate calls so that
        # MAF's internal asyncio.Lock objects stay on one loop.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

    # ------------------------------------------------------------------
    # Agent construction (lazy, so the import is deferred)
    # ------------------------------------------------------------------

    def _build_agent(self):
        """Construct the MAF ``Agent`` on first use."""
        try:
            from agent_framework import Agent
        except ImportError:
            raise ImportError(
                'Microsoft Agent Framework is not installed.  '
                'Install it with: pip install agent-framework --pre'
            )

        client = self._build_client()
        kwargs = dict(
            client=client,
            name=self.agent_name,
            instructions=self.agent_instructions,
        )
        if self.tools:
            kwargs['tools'] = self.tools
        self._agent = Agent(**kwargs)

    def _build_client(self):
        """Return the appropriate MAF chat client."""
        if self.client_type == 'openai':
            from agent_framework.openai import OpenAIChatClient

            api_key = self._resolve_key('OPENAI_API_KEY')
            kwargs: Dict = dict(model_id=self.path)
            if api_key:
                kwargs['api_key'] = api_key
            return OpenAIChatClient(**kwargs)

        elif self.client_type == 'azure_openai':
            from agent_framework.azure import AzureOpenAIChatClient

            api_key = self._resolve_key('AZURE_OPENAI_API_KEY')
            kwargs = dict(deployment_name=self.deployment_name)
            if self.endpoint:
                kwargs['endpoint'] = self.endpoint
            if api_key:
                kwargs['api_key'] = api_key
            else:
                kwargs['credential'] = self._resolve_credential()
            if self.api_version:
                kwargs['api_version'] = self.api_version
            return AzureOpenAIChatClient(**kwargs)

        elif self.client_type == 'azure_responses':
            from agent_framework.azure import AzureOpenAIResponsesClient

            api_key = self._resolve_key('AZURE_OPENAI_API_KEY')
            kwargs = dict(deployment_name=self.deployment_name)
            if self.project_endpoint:
                kwargs['project_endpoint'] = self.project_endpoint
            elif self.endpoint:
                kwargs['endpoint'] = self.endpoint
            if api_key:
                kwargs['api_key'] = api_key
            else:
                kwargs['credential'] = self._resolve_credential()
            if self.api_version:
                kwargs['api_version'] = self.api_version
            return AzureOpenAIResponsesClient(**kwargs)

        elif self.client_type == 'azure_ai':
            from agent_framework.azure_ai import AzureAIChatClient

            api_key = self._resolve_key('AZURE_AI_API_KEY')
            kwargs = dict(model_id=self.deployment_name)
            if self.project_endpoint:
                kwargs['project_endpoint'] = self.project_endpoint
            elif self.endpoint:
                kwargs['endpoint'] = self.endpoint
            if api_key:
                kwargs['api_key'] = api_key
            else:
                kwargs['credential'] = self._resolve_credential()
            return AzureAIChatClient(**kwargs)

        else:
            raise ValueError(
                f'Unsupported client_type={self.client_type!r}.  '
                "Choose from: 'openai', 'azure_openai', "
                "'azure_responses', 'azure_ai'."
            )

    def _resolve_key(self, env_var: str) -> Optional[str]:
        """Return the API key, reading from *env_var* when ``api_key='ENV'``."""
        import os

        if self.api_key == 'ENV':
            return os.environ.get(env_var)
        return self.api_key

    def _resolve_credential(self):
        """Return an Azure ``TokenCredential``.

        Uses the explicitly provided ``self.credential`` if available,
        otherwise falls back to ``DefaultAzureCredential`` from the
        ``azure-identity`` package.
        """
        if self.credential is not None:
            return self.credential

        try:
            from azure.identity import DefaultAzureCredential
        except ImportError:
            raise ImportError(
                'azure-identity is required for Azure Managed Identity '
                'authentication.  Install it with: '
                'pip install azure-identity'
            )

        self.logger.info(
            'No API key found; falling back to '
            'DefaultAzureCredential for authentication.'
        )
        return DefaultAzureCredential()

    # ------------------------------------------------------------------
    # OpenCompass interface
    # ------------------------------------------------------------------

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[PromptType]): A list of strings or PromptDicts.
            max_out_len (int): The maximum length of the output.
            temperature (float): Sampling temperature (unused directly by
                the MAF ``Agent.run`` API but kept for interface
                compatibility).

        Returns:
            List[str]: A list of generated strings.
        """
        if self._agent is None:
            self._build_agent()

        if len(inputs) == 1:
            return [self._generate(inputs[0], max_out_len)]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(
                executor.map(
                    self._generate,
                    inputs,
                    [max_out_len] * len(inputs),
                ))
        return results

    def _generate(self, input: PromptType, max_out_len: int = 512) -> str:
        """Run the MAF agent on a single input with retry logic.

        Args:
            input (PromptType): A string or PromptList.
            max_out_len (int): Maximum output length (forwarded as a hint
                when the agent supports it).

        Returns:
            str: The generated string.
        """
        assert isinstance(input, (str, PromptList))

        prompt = self._prompt_to_str(input)

        num_retries = 0
        while num_retries < self.retry:
            self.wait()
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._agent.run(prompt), self._loop)
                result = future.result()
                return str(result).strip()
            except Exception as e:
                self.logger.error(
                    'MAF agent call failed (attempt %d/%d): %s',
                    num_retries + 1, self.retry, e,
                )
            num_retries += 1

        raise RuntimeError(
            f'Calling MAF agent failed after retrying for '
            f'{self.retry} times. Check the logs for details.'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prompt_to_str(input: PromptType) -> str:
        """Flatten an OpenCompass ``PromptType`` into a plain string."""
        if isinstance(input, str):
            return input

        # PromptList – concatenate role/prompt pairs into a chat-style string
        parts: List[str] = []
        for item in input:
            role = item.get('role', 'HUMAN')
            content = item.get('prompt', '')
            if role == 'SYSTEM':
                parts.append(f'[System]: {content}')
            elif role == 'BOT':
                parts.append(f'[Assistant]: {content}')
            else:
                parts.append(f'[User]: {content}')
        return '\n'.join(parts)
