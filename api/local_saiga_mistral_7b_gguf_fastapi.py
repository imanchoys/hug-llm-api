from fastapi.responses import JSONResponse
from llama_cpp import Llama
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Response
from utils_saiga_mistral_gguf import (
    get_system_tokens,
    get_message_tokens,
    BOT_TOKEN,
    LINE_BREAK_TOKEN
)

# This demo works with llama.cpp python bindings >= 0.1.79 - it expects model GGUF format:
#   - to install: `pip install llama-cpp-python`
MODEL_VAR = "model-q4_K.gguf"
MODEL_URL = f"https://huggingface.co/IlyaGusev/saiga_mistral_7b_gguf/resolve/main/{MODEL_VAR}"
MODEL_DIR = "model_dir"

# path for .gguf model file (could be either run on CPUs or GPUs)
PATH_TO_GGUF = "./models_dir/saiga_mistral_7b/model-q4_K.gguf"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Define a class to hold the conversation state
class ConversationSession:
    def __init__(self):
        self.model: Llama | None = None
        self.tokens: list[int] = []
        self.system_tokens: list[int] = []


# Create a dictionary to store conversation sessions
sessions: dict[str, ConversationSession] = {}


# Endpoint to start a new conversation session
@app.post("/start_session/{session_id}")
def start_session(session_id: str):
    sessions[session_id] = ConversationSession()
    return {"message": f"Session {session_id} started."}


# Endpoint to generate a response for a given session and prompt
@app.post("/generate/{session_id}")
def generate(
    session_id: str,
    prompt: str,
    top_k: int = 30,
    top_p: float = 0.9,
    temperature: float = 0.2,
    repeat_penalty: float = 1.1,
    print_tokens: bool = False
) -> dict:
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )

    session = sessions[session_id]
    if session.model is None:
        session.model = Llama(
            PATH_TO_GGUF,
            n_ctx=2000,
            n_parts=1,
        )

        # TODO: customize "system tokens" creation
        sys_tokens = get_system_tokens(session.model)
        # save system tokens to session
        session.system_tokens = sys_tokens
        # for now tokens is equal to system tokens - so we save them as well
        session.tokens = sys_tokens
        # give the system tokens to model
        session.model.eval(session.tokens)

    res = []
    # create tokens compatible with this model
    # (just manually inserting a few constants defined earlier, like:
    # USER_TOKEN, BOT_TOKEN, SYSTEM_TOKEN, SYSTEM_PROMPT)
    message_tokens = get_message_tokens(
        model=session.model,
        role="user",
        content=prompt
    )

    # tokenization step
    role_tokens = [
        session.model.token_bos(),
        BOT_TOKEN,
        LINE_BREAK_TOKEN
    ]

    session.tokens += message_tokens + role_tokens
    if print_tokens:
        print(session.tokens)

    full_prompt = session.model.detokenize(session.tokens)
    if print_tokens:
        print(session.model.tokenize(full_prompt))

    # generation step = actual use LLM
    generator = session.model.generate(
        session.tokens,
        top_k=top_k,
        top_p=top_p,
        temp=temperature,
        repeat_penalty=repeat_penalty
    )

    print("Generation finished")

    for token in generator:
        token_str = session.model.detokenize([token]).decode(
            "utf-8",
            errors="ignore"
        )

        session.tokens.append(token)
        if token == session.model.token_eos():
            break
        res.append(token_str)

    return {
        "prompt": prompt,
        "result": "".join(res),
        "model": MODEL_URL
    }
