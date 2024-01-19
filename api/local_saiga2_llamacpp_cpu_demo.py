import gradio as gr
import os.path
import gc

from uuid import uuid4
from huggingface_hub import snapshot_download
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    # PDFMinerLoader,
    PyPDFium2Loader,
    TextLoader,
    # UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from chromadb.config import Settings
from llama_cpp import Llama


CONTEXT_SIZE = 2000
ENABLE_GPU = False
GPU_LAYERS = 20

SYSTEM_PROMPT = \
    "Ты - полезный, уважительный и честный ассистент. Всегда отвечай как можно более надежно. "\
    "В ответах не должно информации из твоей базы знаний, а только лишь информация из контекста и ее перефразирование. "\
    "Если вопрос не имеет смысла или не является фактологически последовательным, объясни почему, а не отвечайте на вопрос некорректно. "\
    "Если ты не знаешь ответа на вопрос, пожалуйста, не сообщай ложную информацию. Твоя цель - дать ответы, связанные с базой знаний компании."

SYSTEM_TOKEN = 1788
USER_TOKEN = 1404
BOT_TOKEN = 9225
LINEBREAK_TOKEN = 13

ROLE_TOKENS = {
    "user": USER_TOKEN,
    "bot": BOT_TOKEN,
    "system": SYSTEM_TOKEN
}

LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    # ".pdf": (PDFMinerLoader, {}),
    ".pdf": (PyPDFium2Loader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

REPO_NAME = "IlyaGusev/saiga2_13b_gguf"
MODEL_NAME = "model-q4_K.gguf"
EMBEDDING_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
LOCAL_DIR = "."

if os.path.isdir("/data"):
    print("Persistent storage enabled")

model_path = snapshot_download(
    repo_id=REPO_NAME,
    allow_patterns=MODEL_NAME) + "/" + MODEL_NAME

model = None


def init_model(context_size, enable_gpu=False, gpu_layer_number=35):
    global model

    if model is not None:
        del model
        gc.collect()

    if enable_gpu:
        model = Llama(
            model_path=model_path,
            n_ctx=context_size,
            n_parts=1,
            # n_batch=100,
            logits_all=True,
            # n_threads=12,
            verbose=True,
            n_gpu_layers=gpu_layer_number,
            # n_gqa=8   # must be set for 70b models
        )
        return model
    else:
        model = Llama(
            model_path=model_path,
            n_ctx=context_size,
            n_parts=1,
            # n_batch=100,
            logits_all=True,
            # n_threads=12,
            verbose=True,
            # n_gqa=8   # must be set for 70b models
        )
        return model


init_model(CONTEXT_SIZE, ENABLE_GPU, GPU_LAYERS)

max_new_tokens = 1500
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def get_uuid():
    return str(uuid4())


def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    assert ext in LOADER_MAPPING
    loader_class, loader_args = LOADER_MAPPING[ext]
    loader = loader_class(file_path, **loader_args)
    print("load_single documernt, return type is: ", type(loader.load()[0]))
    return loader.load()[0]


def get_message_tokens(model, role, content):
    message_tokens = model.tokenize(content.encode("utf-8"))
    message_tokens.insert(1, ROLE_TOKENS[role])
    message_tokens.insert(2, LINEBREAK_TOKEN)
    message_tokens.append(model.token_eos())
    return message_tokens


def get_system_tokens(model):
    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    return get_message_tokens(model, **system_message)


def upload_files(files, file_paths):
    file_paths = [f.name for f in files]
    return file_paths


def process_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if len(line.strip()) > 2]
    text = "\n".join(lines).strip()
    if len(text) < 10:
        return None
    return text


def build_index(file_paths, db, chunk_size, chunk_overlap, file_warning):
    documents = [load_single_document(path) for path in file_paths]
    print("build_index, documents type is :", type(documents))
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(documents)
    fixed_documents = []
    for doc in documents:
        doc.page_content = process_text(doc.page_content)
        if not doc.page_content:
            continue
        fixed_documents.append(doc)

    db = Chroma.from_documents(
        fixed_documents,
        embeddings,
        client_settings=Settings(
            anonymized_telemetry=False
        )
    )
    file_warning = f"Загружено {len(fixed_documents)} фрагментов! Можно задавать вопросы."
    return db, file_warning


def user(message, history, system_prompt):
    new_history = history + [[message, None]]
    return "", new_history


def retrieve(history, db, retrieved_docs, k_documents):
    context = ""
    if db:
        last_user_message = history[-1][0]
        retriever = db.as_retriever(search_kwargs={"k": k_documents})
        docs = retriever.get_relevant_documents(last_user_message)
        retrieved_docs = "\n\n".join([doc.page_content for doc in docs])
    return retrieved_docs


def bot(
    history,
    system_prompt,
    conversation_id,
    retrieved_docs,
    top_p,
    top_k,
    temp
):
    if not history:
        return

    tokens = get_system_tokens(model)[:]
    tokens.append(LINEBREAK_TOKEN)

    for user_message, bot_message in history[:-1]:
        message_tokens = get_message_tokens(
            model=model, role="user", content=user_message)
        tokens.extend(message_tokens)
        if bot_message:
            message_tokens = get_message_tokens(
                model=model, role="bot", content=bot_message)
            tokens.extend(message_tokens)

    last_user_message = history[-1][0]
    if retrieved_docs:
        last_user_message = f"Контекст: {retrieved_docs}\n\nИспользуя контекст, ответь на вопрос: {last_user_message}. Не используй свои знания при ответе на вопрос."
    message_tokens = get_message_tokens(
        model=model, role="user", content=last_user_message)
    tokens.extend(message_tokens)

    role_tokens = [model.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
    tokens.extend(role_tokens)
    generator = model.generate(
        tokens,
        top_k=top_k,
        top_p=top_p,
        temp=temp
    )

    partial_text = ""
    for i, token in enumerate(generator):
        if token == model.token_eos() or (max_new_tokens is not None and i >= max_new_tokens):
            break
        partial_text += model.detokenize([token]).decode("utf-8", "ignore")
        history[-1][1] = partial_text
        yield history


with gr.Blocks(
    theme=gr.themes.Soft()
) as demo:
    db = gr.State(None)
    conversation_id = gr.State(get_uuid)
    gr.Markdown(
        """<h1><center>Saiga2 RAG: retrieval QA</center></h1>
        """
        """<h3><center>Credits to Ilya Gusev</center></h1>
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            file_output = gr.File(file_count="multiple",
                                  label="Загрузка файлов")
            file_paths = gr.State([])
            file_warning = gr.Markdown(f"Фрагменты ещё не загружены!")

        with gr.Column(min_width=200, scale=3):
            with gr.Tab(label="Параметры нарезки"):
                chunk_size = gr.Slider(
                    minimum=50,
                    maximum=2000,
                    value=250,
                    step=50,
                    interactive=True,
                    label="Размер фрагментов",
                )
                chunk_overlap = gr.Slider(
                    minimum=0,
                    maximum=500,
                    value=30,
                    step=10,
                    interactive=True,
                    label="Пересечение"
                )

    with gr.Row():
        k_documents = gr.Slider(
            minimum=1,
            maximum=10,
            value=2,
            step=1,
            interactive=True,
            label="Кол-во фрагментов для контекста"
        )
    with gr.Row():
        retrieved_docs = gr.Textbox(
            lines=6,
            label="Извлеченные фрагменты",
            placeholder="Появятся после задавания вопросов",
            interactive=False
        )
    with gr.Row():
        with gr.Column(scale=5):
            system_prompt = gr.Textbox(
                label="Системный промпт", placeholder="", value=SYSTEM_PROMPT, interactive=False)
            chatbot = gr.Chatbot(label="Диалог").style(height=400)
        with gr.Column(min_width=80, scale=1):
            with gr.Tab(label="Параметры генерации"):
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    interactive=True,
                    label="Top-p",
                )
                top_k = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=30,
                    step=5,
                    interactive=True,
                    label="Top-k",
                )
                temp = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.1,
                    step=0.1,
                    interactive=True,
                    label="Temp"
                )

    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(
                label="Отправить сообщение",
                placeholder="Отправить сообщение",
                show_label=False,
            ).style(container=False)
        with gr.Column():
            with gr.Row():
                submit = gr.Button("Отправить")
                stop = gr.Button("Остановить")
                clear = gr.Button("Очистить")

    # Upload files
    upload_event = file_output.change(
        fn=upload_files,
        inputs=[file_output, file_paths],
        outputs=[file_paths],
        queue=True,
    ).success(
        fn=build_index,
        inputs=[file_paths, db, chunk_size, chunk_overlap, file_warning],
        outputs=[db, file_warning],
        queue=True
    )

    # Pressing Enter
    submit_event = msg.submit(
        fn=user,
        inputs=[msg, chatbot, system_prompt],
        outputs=[msg, chatbot],
        queue=False,
    ).success(
        fn=retrieve,
        inputs=[chatbot, db, retrieved_docs, k_documents],
        outputs=[retrieved_docs],
        queue=True,
    ).success(
        fn=bot,
        inputs=[
            chatbot,
            system_prompt,
            conversation_id,
            retrieved_docs,
            top_p,
            top_k,
            temp
        ],
        outputs=chatbot,
        queue=True,
    )

    # Pressing the button
    submit_click_event = submit.click(
        fn=user,
        inputs=[msg, chatbot, system_prompt],
        outputs=[msg, chatbot],
        queue=False,
    ).success(
        fn=retrieve,
        inputs=[chatbot, db, retrieved_docs, k_documents],
        outputs=[retrieved_docs],
        queue=True,
    ).success(
        fn=bot,
        inputs=[
            chatbot,
            system_prompt,
            conversation_id,
            retrieved_docs,
            top_p,
            top_k,
            temp
        ],
        outputs=chatbot,
        queue=True,
    )

    # Stop generation
    stop.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[submit_event, submit_click_event],
        queue=False,
    )

    # Clear history
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue(max_size=128, concurrency_count=1)
demo.launch()
