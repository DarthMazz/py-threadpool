import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

import boto3
from botocore.config import Config

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

custom_config = Config(
    retries={
        # 'max_attempts': 5,  # 必要に応じてデフォルトの3から変更
        'mode': 'standard'
    },
    connect_timeout=60,
    read_timeout=60,
)

# Bedrockクライアントの初期化
# 必要に応じてregion_nameやその他の設定を調整してください
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='ap-northeast-1', # 例: Bedrockが利用可能なリージョンを指定
    config=custom_config
)

def translate_text(text: str, source_lang: str, target_lang: str, model_id: str = "apac.amazon.nova-pro-v1:0") -> str:
    """
    指定されたテキストをAWS Bedrock (Titan Text Express) を使用して翻訳します。
    """
    prompt = f"Translate the following {source_lang} text to {target_lang}: {text}"
    
    body = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "inferenceConfig": {
            "max_new_tokens": 4096,
        }
    })

    try:
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response.get('body').read())
        translated_text = response_body['output']['message']['content']
        logging.info(f"Successfully translated text: '{text[:30]}...' -> '{translated_text[:30]}...'")
        return translated_text
    except Exception as e:
        logging.error(f"Error translating text: '{text[:50]}...' - {e}")
        return f"Translation error: {e}"

def parallel_translate(texts: list[str], source_lang: str, target_lang: str, max_workers: int = 5) -> list[str]:
    """
    複数のテキストを並列で翻訳します。
    """
    translated_results = []
    
    # ThreadPoolExecutor を使用して並列実行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # map を使うと順序が保持されますが、as_completed を使うと完了した順に処理できます。
        # 今回は、タスクと元のテキストの対応付けを容易にするため、as_completed と辞書を使います。
        future_to_text = {executor.submit(translate_text, text, source_lang, target_lang): text for text in texts}
        
        logging.info(f"Submitting {len(texts)} translation tasks with {max_workers} workers.")
        
        for future in as_completed(future_to_text):
            original_text = future_to_text[future]
            try:
                translated_text = future.result()
                translated_results.append(translated_text)
            except Exception as exc:
                logging.error(f"Text '{original_text[:50]}...' generated an exception: {exc}")
                translated_results.append(f"Error for '{original_text}': {exc}")
    
    return translated_results

if __name__ == "__main__":
    start_time = time.perf_counter() # 処理開始時刻

    texts_to_translate = [
        "Hello, how are you?",
        "This is a test sentence for parallel translation.",
        "Python is a versatile programming language.",
        "AWS Bedrock allows access to various foundation models.",
        "Parallel execution can significantly speed up I/O bound tasks.",
        "In this example, we are using the Amazon Titan Text Express model.",
        "日本語のテキストも翻訳できるか試してみます。",
        "これは並列処理の効率を示すためのものです。",
        "クラウドサービスは現代のソフトウェア開発に不可欠です。",
        "機械学習モデルは日々進化しています。",
    ]

    source_language = "en"
    target_language = "ja"
    
    logging.info(f"Starting parallel translation from {source_language} to {target_language} for {len(texts_to_translate)} texts.")
    
    # 並列ワーカー数を調整して試してみてください
    # Bedrockのスループット制限やネットワーク帯域も考慮に入れる必要があります
    num_workers = 10
    
    translated_sentences = parallel_translate(texts_to_translate, source_language, target_language, max_workers=num_workers)

    logging.info("\n--- Translated Results ---")
    for i, (original, translated) in enumerate(zip(texts_to_translate, translated_sentences)):
        logging.info(f"Original {i+1}: {original}")
        logging.info(f"Translated {i+1}: {translated}\n")

    # 日本語から英語への翻訳例
    japanese_texts = [
        "こんにちは、元気ですか？",
        "これは並列翻訳のためのテスト文です。",
        "Pythonは多機能なプログラミング言語です。"
    ]
    logging.info(f"Starting parallel translation from ja to en for {len(japanese_texts)} texts.")
    translated_japanese_sentences = parallel_translate(japanese_texts, "ja", "en", max_workers=5)
    
    logging.info("\n--- Japanese to English Translated Results ---")
    for i, (original, translated) in enumerate(zip(japanese_texts, translated_japanese_sentences)):
        logging.info(f"Original {i+1}: {original}")
        logging.info(f"Translated {i+1}: {translated}\n")
    
    end_time = time.perf_counter()   # 処理終了時刻
    elapsed_time = end_time - start_time
    logging.info(f"処理時間 (perf_counter): {elapsed_time:.6f}秒")