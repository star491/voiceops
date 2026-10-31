"""Demo UI (for the 2-minute video): talk to the agent with your microphone, or type.

    python app.py      -> open http://127.0.0.1:7860
"""
import time
from pathlib import Path

import gradio as gr

from src.agent import Agent
from src.kb import KnowledgeBase
from src.llm import DailyLimitReached
from src.voice import voice_turn

KB = KnowledgeBase()
OUT = Path("demo_audio")
OUT.mkdir(exist_ok=True)


def new_agent():
    return Agent("rag", kb=KB)


def _fmt(r):
    tools = "\n".join(f"- `{c['tool']}` {c['args']} → {str(c['result'])[:200]}" for c in r["tool_calls"]) or "none"
    chunks = "\n\n".join(f"**[{k}]** {v[:400]}" for k, v in r["chunk_texts"].items())
    return f"**Chunks used:** {', '.join(r['used_chunks']) or '—'}\n\n**Tool calls:**\n{tools}\n\n---\n{chunks}"


def on_audio(audio_path, chat, agent):
    if not audio_path:
        return chat, None, gr.update(), agent
    try:
        r = voice_turn(audio_path, agent, out_file=OUT / f"reply_{int(time.time() * 1000)}.wav")
    except DailyLimitReached as ex:
        return chat + [{"role": "assistant", "content": f"(LLM daily limit reached on all models: {ex})"}], None, gr.update(), agent
    chat = chat + [{"role": "user", "content": f"🎙️ {r['heard'] or '(unclear audio)'}"},
                   {"role": "assistant", "content": r["reply"]}]
    return chat, r["audio_out"], _fmt(r), agent


def on_text(text, chat, agent):
    if not text.strip():
        return chat, "", gr.update(), agent
    try:
        r = agent.turn(text)
    except DailyLimitReached as ex:
        return chat + [{"role": "assistant", "content": f"(LLM daily limit reached on all models: {ex})"}], "", gr.update(), agent
    chat = chat + [{"role": "user", "content": text}, {"role": "assistant", "content": r["reply"]}]
    return chat, "", _fmt(r), agent


with gr.Blocks(title="ZentroMart VoiceOps") as demo:
    gr.Markdown("## ZentroMart voice support agent · Gnani STT/TTS + RAG")
    agent = gr.State(new_agent)
    with gr.Row():
        with gr.Column(scale=3):
            try:   # Gradio 4/5 need type="messages"; newer versions use messages by default
                chat = gr.Chatbot(type="messages", height=420)
            except TypeError:
                chat = gr.Chatbot(height=420)
            mic = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speak to the agent")
            reply_audio = gr.Audio(label="Agent reply", autoplay=True)
            txt = gr.Textbox(placeholder="…or type a message and press Enter")
            reset = gr.Button("New call")
        with gr.Column(scale=2):
            trace = gr.Markdown("Retrieved chunks and tool calls appear here.")
    mic.stop_recording(on_audio, [mic, chat, agent], [chat, reply_audio, trace, agent])
    mic.upload(on_audio, [mic, chat, agent], [chat, reply_audio, trace, agent])
    txt.submit(on_text, [txt, chat, agent], [chat, txt, trace, agent])
    reset.click(lambda: ([], None, "New call started.", new_agent()), None, [chat, reply_audio, trace, agent])

if __name__ == "__main__":
    demo.launch(inbrowser=True)   # opens the page in your browser automatically
