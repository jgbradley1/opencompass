from opencompass.models import MAFAgent

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
)

# ------------------------------------------------------------------ #
# Microsoft Agent Framework – OpenAI provider
# ------------------------------------------------------------------ #
# Set the OPENAI_API_KEY environment variable before running.

models = [
    dict(
        abbr='maf-agent-gpt-4o-mini',
        type=MAFAgent,
        path='gpt-4o-mini',
        client_type='openai',
        agent_name='OpenCompassAgent',
        agent_instructions='You are a helpful assistant.',
        api_key='ENV',
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=4096,
        batch_size=8,
    ),
]
