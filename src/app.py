import gradio as gr
import whisper
import torch
import logging
import warnings

# FutureWarningの一時的な抑制（オプション）
warnings.filterwarnings("ignore", category=FutureWarning)

# ロギングの設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# モデルの読み込み
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    whisper_model = whisper.load_model("large-v3-turbo").to(device)
    logging.info("Whisperモデルを正常にロードしました。")
except Exception as e:
    logging.error(f"モデルのロード中にエラーが発生しました: {str(e)}")
    raise e


def transcribe(audio):
    if audio is None:
        logging.warning("音声ファイルがアップロードされていません。")
        return "音声ファイルがアップロードされていません。"
    try:
        logging.info("文字起こしを開始します。")
        result = whisper_model.transcribe(audio, language="ja")
        text = result["text"]
        logging.info("文字起こしが完了しました。")
    except Exception as e:
        logging.error(f"文字起こし中にエラーが発生しました: {str(e)}")
        text = f"文字起こし中にエラーが発生しました: {str(e)}"

    return text


# Gradioのインターフェースの設定
iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.Textbox(label="文字起こし結果"),
    title="wisper音声認識アプリ",
    description="音声ファイルをアップロードして文字起こしを行います。",
    examples=[["data/o1short.mp3"]],
)

if __name__ == "__main__":
    # アプリの実行
    iface.launch()
