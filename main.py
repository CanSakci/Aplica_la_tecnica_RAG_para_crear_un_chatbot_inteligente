import os
from dotenv import load_dotenv
from core.rag_system import RAGSystem
from core.chatbot import Chatbot

def main():
    # Cargar variables de entorno (API key)
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error crítico: OPENAI_API_KEY no encontrada en el archivo .env")
        return

    print("Iniciando sistema... Cargando y vectorizando documentos markdown...")
    
    try:
        # 1. Inicializar sistema RAG
        rag = RAGSystem()
        
        # 2. Procesar documentos
        doc_paths = ["documents/documento1.md", "documents/documento2.md"]
        rag.process_documents(doc_paths)
        print("¡Sistema de Embeddings inicializado correctamente!")
        
        # 3. Inicializar Chatbot
        chatbot = Chatbot(rag.get_retriever())
        
    except Exception as e:
        print(f"Error al inicializar el sistema: {e}")
        return

    print("\n" + "="*50)
    print("🤖 Chatbot IA de Empresa Listo")
    print("Escribe '/salir' o 'quit' para terminar la conversación.")
    print("="*50 + "\n")

    # Bucle principal de la interfaz CLI
    while True:
        try:
            user_input = input("Tú: ")
            
            # Comando de salida
            if user_input.lower() in ['/salir', 'quit']:
                print("Chatbot: ¡Hasta luego! Ha sido un placer ayudarte.")
                break
                
            if not user_input.strip():
                continue
                
            # Generar respuesta
            response = chatbot.ask(user_input)
            print(f"Chatbot: {response}\n")
            
        except Exception as e:
            # Manejo de errores básicos (conexión, límite de tokens, etc.)
            print(f"\n[Error de sistema]: No se pudo procesar la respuesta. Detalles: {e}\n")

if __name__ == "__main__":
    main()