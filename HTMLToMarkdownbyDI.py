
"""

参考にしたコード
https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/documentintelligence/azure-ai-documentintelligence/samples/sample_analyze_layout.py

USAGE:
    Set the environment variables with your own values before running the sample:
    1) DOCUMENTINTELLIGENCE_ENDPOINT - the endpoint to your Document Intelligence resource.
    2) DOCUMENTINTELLIGENCE_API_KEY - your Document Intelligence API key.


あらかじめ、Azure Document Intelligenceのリソースを作成しておく必要があります。
また、pip install azure-ai-documentintelligenceを実行しておく必要があります。
        
"""

import os

def analyze_documents_output_in_markdown():
    # [START analyze_documents_output_in_markdown]
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat, AnalyzeResult

    endpoint = os.environ["DOCUMENTINTELLIGENCE_ENDPOINT"]
    key = os.environ["DOCUMENTINTELLIGENCE_API_KEY"]
    document_intelligence_client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    with open("<用意したHTMLのファイル名>", "rb") as f:
        html_bytes = f.read()
        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout",
            body=AnalyzeDocumentRequest(bytes_source=html_bytes),
           output_content_format=DocumentContentFormat.MARKDOWN,
        )
    result: AnalyzeResult = poller.result()

    print(f"Here's the full content in format {result.content_format}:\n")
    print(result.content)
    # [END analyze_documents_output_in_markdown]
    return result

result=analyze_documents_output_in_markdown()

