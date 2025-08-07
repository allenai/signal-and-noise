import os, datetime, time, concurrent
from tqdm import tqdm

from openai import OpenAI, RateLimitError, InternalServerError
from openai.types.chat import ChatCompletion

from .__init__ import ROOT_DIR

DEFAULT_OPENAI_SECRET_PATH = os.path.join(ROOT_DIR, '.openai')

DEFAULT_KWARGS_OPENAI = {
    'model': "gpt-4o-mini",
    'top_p': 0.9,
    'temperature': 0.9,
    'max_tokens': 256, 
    'frequency_penalty': 0,
    'presence_penalty': 0,
    # 'top_logprobs': 1,
}

# silence the httpx logger
import logging
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)


def openai_init(secret_path=DEFAULT_OPENAI_SECRET_PATH):
    global CLIENT

    if not os.path.exists(secret_path): 
        raise RuntimeError(f'Need an OpenAI Key! Did not find a key at {secret_path}')
    
    with open(secret_path, 'r') as f: 
        api_key = f.read().strip()
    
    CLIENT = OpenAI(api_key=api_key)


def _call_openai(client, prompt: str, params, max_retries=7, base_delay=2) -> ChatCompletion:
    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                messages=[
                    { "role": "user", "content": prompt }
                ],
                **params
            )
            return response
        except (RateLimitError, InternalServerError) as e:
            retries += 1
            print(f"OpenAI API request exceeded rate limit: {e}. Retrying ({retries}/{max_retries})...")
            if retries < max_retries:
                delay = base_delay * (2 ** retries)
                time.sleep(delay)
            else:
                raise RuntimeError('OpenAI API failed to respond')


def generate_gpt(prompt: list[str], parallel: bool=True, **kwargs) -> list[str]:
    """ Genereate with ChatGPT """
    start_time = datetime.datetime.now()

    assert isinstance(prompt, list), f'Prompt must be a list of string: {prompt}'
    
    params = {**DEFAULT_KWARGS_OPENAI, **kwargs}

    print(f'Generating {len(prompt)} examples with params {params}')
    
    if parallel:
        # Query OpenAI using concurrent
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as exec:
        # with concurrent.futures.ProcessPoolExecutor() as exec:
            futures = [exec.submit(_call_openai, CLIENT, p, params) for p in prompt]
            cands = [f.result() for f in tqdm(futures, desc="Querying OpenAI")]
        cands = [c.choices[0].message.content for c in cands]
    else:
        # Query OpenAI sequentially
        cands = []
        for p in tqdm(prompt, desc="Querying OpenAI"):
            resp: ChatCompletion = _call_openai(CLIENT, p, params)
            cands.append(resp.choices[0].message.content)

    duration = (datetime.datetime.now() - start_time).total_seconds()
    print(f"Generated {len(prompt)} queries in {duration:.2f}s at {len(prompt)/duration:.2f} prompt/s.")

    return cands


def print_estimate_cost(prompt: list[str], model: str="gpt-4o", input_cost: float=5, output_cost: float=15):
    """
    See: https://openai.com/api/pricing
    """
    from tiktoken import encoding_for_model
    enc = encoding_for_model(model)

    total_toks = 0
    for p in prompt:
        encoding = enc.encode(p)
        total_toks += len(encoding)

    input_cost  = (total_toks * (input_cost / 1_000_000))

    # STRONG ASSUMPTION that input length == output length
    # (usually output length is much smaller, and usually "max_token" is set regardless)
    output_cost = (total_toks * (output_cost / 1_000_000)) 

    cost = input_cost + output_cost

    print(f'Cost: ${cost:.4f} on "{model}" for {total_toks} toks.')