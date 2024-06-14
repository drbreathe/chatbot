import os
import pinecone
from dotenv import load_dotenv

# load your credentials from .env file
load_dotenv()

EMBEDDING_DIM = 1536

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index = os.getenv("PINECONE_INDEX_NAME")
pinecone_namespace = os.getenv("PINECONE_NAMESPACE")


def index_delete_all():
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT"),
    )
    index = pinecone.Index(pinecone_index)
    try:
        namespace = pinecone_namespace if pinecone_namespace else ""
        index.delete(delete_all=True, namespace=namespace)
        return "Successfully deleted"
    except:
        # gcp-starter does not allow deleteAll
        # alternatively, you can delete the index and create a new one
        # see: https://docs.pinecone.io/docs/starter-environment
        pinecone.delete_index(pinecone_index)
        pinecone.create_index(pinecone_index, dimension=EMBEDDING_DIM)
        return "Successfully deleted"

def main():
    index_delete_all()

if __name__ == '__main__':
    main()