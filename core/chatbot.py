from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

class Chatbot:
    def __init__(self, retriever):
        # Utiliza el modelo gpt-4o requerido
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.chat_history = [] # Para mantener el contexto de la conversación
        
        system_prompt = (
            "Eres un asistente inteligente para una empresa. Responde a las preguntas "
            "basándote ÚNICAMENTE en el siguiente contexto recuperado.\n\n"
            "Contexto: {context}\n\n"
            "Si no sabes la respuesta basándote en el contexto, indica amablemente "
            "que no tienes esa información."
        )
        
        # Creamos el prompt incluyendo el historial para mantener el contexto
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # Cadena para procesar los documentos y generar la respuesta
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        # Cadena RAG completa
        self.rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    def ask(self, query: str) -> str:
        # Ejecutamos la consulta
        response = self.rag_chain.invoke({
            "input": query,
            "chat_history": self.chat_history
        })
        
        answer = response["answer"]
        
        # Actualizamos el historial de conversación
        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=answer))
        
        return answer