
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.models import ModelFamily, ModelInfo

# https://github.com/microsoft/autogen/blob/main/python/packages/autogen-ext/src/autogen_ext/models/ollama/_model_info.pyに含まれていないモデルを追加する場合は、設定が必要
model_info={                       # 必須のモデル情報
    "vision": False,               # ビジョン機能の有無
    "function_calling": True,      # 関数呼び出し機能の有無
    "json_output": True,           # JSON 出力対応
    "family": ModelFamily.UNKNOWN,
    "structured_output": True,     # 構造化出力対応
}


ollama_model_client = OllamaChatCompletionClient(model="<利用するモデル>",model_info=model_info)

firstAgent = AssistantAgent(
    "first",
    system_message="一人目の役割定義",
    model_client=ollama_model_client
)

secondAgent = AssistantAgent(
    "second",
    system_message="二人目の役割定義",
    model_client=ollama_model_client
)

from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination

# 仮に6対話で終わる設定
max_message_termination =  MaxMessageTermination(6)

# Create a team with the primary and critic agents.
team = RoundRobinGroupChat([firstAgent, secondAgent], termination_condition=max_message_termination)
result = await team.run(task="タスクをここに記入")

for message in result.messages:
    print(message)