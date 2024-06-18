
from langchain.agents import ConversationalChatAgent


def chatbot(session_id, user_message):
    # search = SerpAPIWrapper()
    # tools = [
    #      Tool(
    #     name="Search",
    #     func=search.run,
    #     description="useful for when you need to answer questions about current events."
    #     )
    # ]    

    llm = ChatOpenAI(temperature=1, max_tokens=3000)

    message_history = MongoDBChatMessageHistory(
    connection_string=connectionString , session_id=session_id, database_name = dbName, collection_name = 'message_stores'
    )

    memory = ConversationBufferWindowMemory(
    memory_key="chat_history", chat_memory=message_history, llm=llm
    , return_messages=True,k= 0
    )

    system_prompt_template = "You are a nice chatbot designed to provide creative ideas for YouTube video content. Your main goal is to consistently provide content ideas for YouTube video creation. Always generate top four to five content ideas for the video related to - {input} except when user input is a greeting like hi, hello or farewell like bye, goodbye. Ensure your responses are innovative, valuable, and align with YouTube's community guidelines. Avoid offensive ideas. Aim to reply in 50 words, and capable of handling simple interactions like greetings and farewells."


    custom_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm,  system_message=system_prompt_template) 
    agent_executor = AgentExecutor.from_agent_and_tools(agent=custom_agent, memory=memory,verbose = True)


    try:
    
        response = agent_executor.run(user_message)
    
        return response
    
    except OutputParserException as e:
        message = str(e)
        if "{" in message:
        
            return 'Kindly provide the correct topics, keywords, or queries. Thank you.'

        else:
            message = message.replace("Could not parse LLM output: ", " ")

            return message


@app.route('/ai/api/chatbot', methods=['POST'])
def chatbot_api():
    data = request.get_json()

    
    user_message = data.get('user_message', '')
    session_id = data.get('session_id', '')
    clear_memory = data.get('clear_memory', '')
    
    # token = request.headers.get('Authorization').split()[1]
    # auth = secure_endpoint(token)
    # success = auth['success']
    # userId = auth['userId']
    # status = auth['status']
    # if success == False:
    #     auth.pop('userId')
    #     return jsonify(auth)
    
    try:
        if clear_memory==True:
            clear_status = clear_chat_history(session_id)
            # return jsonify({'success': success, 'status': status})
        else:
            input_string = user_message
            input_string = ''.join(input_string.split())
            
            if not input_string or input_string.isdigit():
                return jsonify({"AI": "Please provide topics, keywords, or queries for content ideas."})
                # return jsonify({"AI": "Please provide topics, keywords, or queries for content ideas.", 'success': success, 'userId': userId, 'status': status})
                
    
            else:
                bot_reply=chatbot(session_id, user_message)
                return jsonify({"AI": bot_reply})
                # return jsonify({"AI": bot_reply, 'success': success, 'userId': userId, 'status': status})
                
    except Exception as e:
        
        print("Exception",e)
