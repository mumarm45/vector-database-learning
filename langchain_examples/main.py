from modules.embeddings_model import call_retriever

def main():
    result = call_retriever("email policy", k=5)
    print(result)


if __name__ == "__main__":
    main()
