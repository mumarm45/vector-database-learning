
from modules.create_chroma import initiate_food_collection
from modules.llm_food_rag_search import start_chat

def main():
    collection = initiate_food_collection()
    # result = perform_similarity_search(collection, "find me a food that is a pasta")
    # result = perform_filtered_similarity_search(collection, 
    # "find best pizza", cuisine_filter="Italian", max_calories=500)
    # interactive_food_chatbot(collection)
    # print(result)
    # interactive_advanced_search(collection)
    start_chat(collection)



if __name__ == "__main__":
    main()
