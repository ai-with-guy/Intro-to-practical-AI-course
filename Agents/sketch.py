class Agent:
    def __init__(self):
        self.messages = [
            {"system": "You are an AI agent with tools ..."}
        ]

    def respond(self, message):
        display(message)
        self.messages.append(message)

        response = api_call(self.messages)
        display(response)

        self.messages.append(response)

        #check using the models specific tokenizer what tool calls are made
        if response.is_tool_call():
            message2 = response.do_tool()
            self.respond(message2)