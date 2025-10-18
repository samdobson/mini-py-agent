import os
import json
from anthropic import Anthropic


def edit_file(inputs):
    """Edit a file by replacing old_str with new_str."""
    path = inputs.get("path")
    old_str = inputs.get("old_str")
    new_str = inputs.get("new_str")

    if not path or old_str == new_str:
        raise ValueError("Invalid input parameters")

    try:
        with open(path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        if old_str == "":
            if os.path.dirname(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(new_str)
            return f"Successfully created file {path}"
        raise

    new_content = content.replace(old_str, new_str)

    if content == new_content and old_str != "":
        raise ValueError("old_str not found in file")

    with open(path, "w") as f:
        f.write(new_content)

    return "OK"


EDIT_FILE_TOOL = {
    "name": "edit_file",
    "description": """Make edits to a text file.

Replaces 'old_str' with 'new_str' in the given file. 'old_str' and 'new_str' MUST be different from each other.

If the file specified with path doesn't exist, it will be created.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "The path to the file"},
            "old_str": {
                "type": "string",
                "description": "Text to search for - must match exactly",
            },
            "new_str": {
                "type": "string",
                "description": "Text to replace old_str with",
            },
        },
        "required": ["path", "old_str", "new_str"],
    },
    "function": edit_file,
}


def list_files(inputs):
    """List files and directories at a given path."""
    path = inputs.get("path", ".")
    files = []
    for root, dirs, filenames in os.walk(path):
        for d in dirs:
            files.append(os.path.relpath(os.path.join(root, d), path) + "/")
        for f in filenames:
            files.append(os.path.relpath(os.path.join(root, f), path))
    return json.dumps(files)


LIST_FILES_TOOL = {
    "name": "list_files",
    "description": "List files and directories at a given path. If no path is provided, lists files in the current directory.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Optional relative path to list files from. Defaults to current directory if not provided.",
            }
        },
    },
    "function": list_files,
}


def read_file(inputs):
    """Read the contents of a file."""
    path = inputs.get("path")
    with open(path, "r") as f:
        return f.read()


READ_FILE_TOOL = {
    "name": "read_file",
    "description": "Read the contents of a given relative file path. Use this when you want to see what's inside a file. Do not use this with directory names.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The relative path of a file in the working directory.",
            }
        },
        "required": ["path"],
    },
    "function": read_file,
}


def main():
    client = Anthropic()

    tools = [READ_FILE_TOOL, LIST_FILES_TOOL, EDIT_FILE_TOOL]
    agent = Agent(client, tools)
    agent.run()


class Agent:
    def __init__(self, client, tools):
        self.client = client
        self.tools = tools

    def run(self):
        conversation = []
        print("Chat with Claude (use 'ctrl-c' to quit)")

        while True:
            print("You: ", end="")
            user_input = input()
            if not user_input.strip():
                continue
            conversation.append({"role": "user", "content": user_input})

            while True:
                message = self._run_inference(conversation)
                conversation.append(message)

                tool_results = []
                for content in message["content"]:
                    if content["type"] == "text":
                        print(f"Claude: {content['text']}")
                    elif content["type"] == "tool_use":
                        result = self._execute_tool(
                            content["id"], content["name"], content["input"]
                        )
                        tool_results.append(result)

                if not tool_results:
                    break

                conversation.append({"role": "user", "content": tool_results})

    def _run_inference(self, conversation):
        tools = [
            {k: tool[k] for k in ["name", "description", "input_schema"]}
            for tool in self.tools
        ]
        response = self.client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            messages=conversation,
            tools=tools if tools else None,
        )

        content = [
            (
                {"type": "text", "text": block.text}
                if block.type == "text"
                else {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
            for block in response.content
        ]

        return {"role": response.role, "content": content}

    def _execute_tool(self, tool_id, name, inputs):
        tool = next((t for t in self.tools if t["name"] == name), None)

        print(f"tool: {name}({inputs})")

        try:
            result = tool["function"](inputs)
            return {"type": "tool_result", "tool_use_id": tool_id, "content": result}
        except Exception as e:
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": str(e),
                "is_error": True,
            }


if __name__ == "__main__":
    main()
