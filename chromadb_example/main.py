
import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.Client()


def create_collection(collection_name): 
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=ef
    )
    return collection

def perform_operations():
    try:
        collection_name = "my_grocery_collection"
        collection = create_collection(collection_name)

        add_data(collection, *get_data())
        print(f"Collection '{collection_name}' created successfully.")
        return collection
    except Exception as error:
        print(f"Error: {error}")

def add_data(collection, texts, ids):
    collection.add(
    documents=texts,
    metadatas=[{"source": "grocery_store", "category": "food"} for _ in texts],
    ids=ids
   )

def get_all_data(collection):
    return collection.get()
def  filter_by_text(collection, query_term): 
    results = collection.query(
            query_texts=[query_term],
            n_results=3,
            include=["metadatas", "documents", "distances"],
        )    
    return results
def perform_similarity_search(collection, query_term):
    try:
        results = filter_by_text(collection, "")
        if not results or not results['ids'] or len(results['ids'][0]) == 0:
            print(f'No documents found similar to "{query_term}"')
            return
        print(f'Top 3 similar documents to "{query_term}":')
        for i in range(min(3, len(results['ids'][0]))):
            doc_id = results['ids'][0][i] 
            score = results['distances'][0][i] 
            text = results['documents'][0][i]
            if not text:
                print(f' - ID: {doc_id}, Text: "Text not available", Score: {score:.4f}')
            else:
                print(f' - ID: {doc_id}, Text: "{text}", Score: {score:.4f}')
    except Exception as error:
        print(f"Error in similarity search: {error}")    

def get_data():
    texts = [
    'fresh red apples',
    'organic bananas',
    'ripe mangoes',
    'whole wheat bread',
    'farm-fresh eggs',
    'natural yogurt',
    'frozen vegetables',
    'grass-fed beef',
    'free-range chicken',
    'fresh salmon fillet',
    'aromatic coffee beans',
    'pure honey',
    'golden apple',
    'red fruit'
    ]
    ids = [f"food_{index + 1}" for index, _ in enumerate(texts)]
    return texts, ids


def employee_data():
    employees = [
            {
                "id": "employee_1",
                "name": "John Doe",
                "experience": 5,
                "department": "Engineering",
                "role": "Software Engineer",
                "skills": "Python, JavaScript, React, Node.js, databases",
                "location": "New York",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_2",
                "name": "Jane Smith",
                "experience": 8,
                "department": "Marketing",
                "role": "Marketing Manager",
                "skills": "Digital marketing, SEO, content strategy, analytics, social media",
                "location": "Los Angeles",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_3",
                "name": "Alice Johnson",
                "experience": 3,
                "department": "HR",
                "role": "HR Coordinator",
                "skills": "Recruitment, employee relations, HR policies, training programs",
                "location": "Chicago",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_4",
                "name": "Michael Brown",
                "experience": 12,
                "department": "Engineering",
                "role": "Senior Software Engineer",
                "skills": "Java, Spring Boot, microservices, cloud architecture, DevOps",
                "location": "San Francisco",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_5",
                "name": "Emily Wilson",
                "experience": 2,
                "department": "Marketing",
                "role": "Marketing Assistant",
                "skills": "Content creation, email marketing, market research, social media management",
                "location": "Austin",
                "employment_type": "Part-time"
            },
            {
                "id": "employee_6",
                "name": "David Lee",
                "experience": 15,
                "department": "Engineering",
                "role": "Engineering Manager",
                "skills": "Team leadership, project management, software architecture, mentoring",
                "location": "Seattle",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_7",
                "name": "Sarah Clark",
                "experience": 8,
                "department": "HR",
                "role": "HR Manager",
                "skills": "Performance management, compensation planning, policy development, conflict resolution",
                "location": "Boston",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_8",
                "name": "Chris Evans",
                "experience": 20,
                "department": "Engineering",
                "role": "Senior Architect",
                "skills": "System design, distributed systems, cloud platforms, technical strategy",
                "location": "New York",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_9",
                "name": "Jessica Taylor",
                "experience": 4,
                "department": "Marketing",
                "role": "Marketing Specialist",
                "skills": "Brand management, advertising campaigns, customer analytics, creative strategy",
                "location": "Miami",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_10",
                "name": "Alex Rodriguez",
                "experience": 18,
                "department": "Engineering",
                "role": "Lead Software Engineer",
                "skills": "Full-stack development, React, Python, machine learning, data science",
                "location": "Denver",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_11",
                "name": "Hannah White",
                "experience": 6,
                "department": "HR",
                "role": "HR Business Partner",
                "skills": "Strategic HR, organizational development, change management, employee engagement",
                "location": "Portland",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_12",
                "name": "Kevin Martinez",
                "experience": 10,
                "department": "Engineering",
                "role": "DevOps Engineer",
                "skills": "Docker, Kubernetes, AWS, CI/CD pipelines, infrastructure automation",
                "location": "Phoenix",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_13",
                "name": "Rachel Brown",
                "experience": 7,
                "department": "Marketing",
                "role": "Marketing Director",
                "skills": "Strategic marketing, team leadership, budget management, campaign optimization",
                "location": "Atlanta",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_14",
                "name": "Matthew Garcia",
                "experience": 3,
                "department": "Engineering",
                "role": "Junior Software Engineer",
                "skills": "JavaScript, HTML/CSS, basic backend development, learning frameworks",
                "location": "Dallas",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_15",
                "name": "Olivia Moore",
                "experience": 12,
                "department": "Engineering",
                "role": "Principal Engineer",
                "skills": "Technical leadership, system architecture, performance optimization, mentoring",
                "location": "San Francisco",
                "employment_type": "Full-time"
            },
        ]
    
    # Create comprehensive text documents for each employee
# These documents will be used for similarity search based on skills, roles, and experience
    
    ids = [employee["id"] for employee in employees]    
    return employees, ids
def employee_documents(employees):
    employee_documents = []
    for employee in employees:
        document = f"{employee['role']} with {employee['experience']} years of experience in {employee['department']}. "
        document += f"Skills: {employee['skills']}. Located in {employee['location']}. "
        document += f"Employment type: {employee['employment_type']}."
        employee_documents.append(document)    
    return employee_documents    

def employee_collection():
    collection_name = "my_employee_collection"
    collection = create_collection(collection_name)
    employees, ids = employee_data()
    collection.add(
    ids=ids,
    documents=employee_documents(employees),
    metadatas=[{
        "name": employee["name"],
        "department": employee["department"],
        "role": employee["role"],
        "experience": employee["experience"],
        "location": employee["location"],
        "employment_type": employee["employment_type"]
    } for employee in employees]
    )
    return collection

def refine_filter(collection, query):
    query_text = query
    results = collection.query(
    query_texts=[query_text],
    n_results=5,
    where={
        "$and": [
            {"experience": {"$gte": 5}},
            {"location": {"$in": ["San Francisco", "New York", "Seattle"]}}
        ]
    }
)

def books_data():
    books =  [
    {
        "id": "book_1",
        "title": "The Great Gatsby",
        "author": "F. Scott Fitzgerald",
        "genre": "Classic",
        "year": 1925,
        "rating": 4.1,
        "pages": 180,
        "description": "A tragic tale of wealth, love, and the American Dream in the Jazz Age",
        "themes": "wealth, corruption, American Dream, social class",
        "setting": "New York, 1920s"
    },
    {
        "id": "book_2",
        "title": "To Kill a Mockingbird",
        "author": "Harper Lee",
        "genre": "Classic",
        "year": 1960,
        "rating": 4.3,
        "pages": 376,
        "description": "A powerful story of racial injustice and moral growth in the American South",
        "themes": "racism, justice, moral courage, childhood innocence",
        "setting": "Alabama, 1930s"
    },
    {
        "id": "book_3",
        "title": "1984",
        "author": "George Orwell",
        "genre": "Dystopian",
        "year": 1949,
        "rating": 4.4,
        "pages": 328,
        "description": "A chilling vision of totalitarian control and surveillance society",
        "themes": "totalitarianism, surveillance, freedom, truth",
        "setting": "Oceania, dystopian future"
    },
    {
        "id": "book_4",
        "title": "Harry Potter and the Philosopher's Stone",
        "author": "J.K. Rowling",
        "genre": "Fantasy",
        "year": 1997,
        "rating": 4.5,
        "pages": 223,
        "description": "A young wizard discovers his magical heritage and begins his education at Hogwarts",
        "themes": "friendship, courage, good vs evil, coming of age",
        "setting": "England, magical world"
    },
    {
        "id": "book_5",
        "title": "The Lord of the Rings",
        "author": "J.R.R. Tolkien",
        "genre": "Fantasy",
        "year": 1954,
        "rating": 4.5,
        "pages": 1216,
        "description": "An epic fantasy quest to destroy a powerful ring and save Middle-earth",
        "themes": "heroism, friendship, good vs evil, power corruption",
        "setting": "Middle-earth, fantasy realm"
    },
    {
        "id": "book_6",
        "title": "The Hitchhiker's Guide to the Galaxy",
        "author": "Douglas Adams",
        "genre": "Science Fiction",
        "year": 1979,
        "rating": 4.2,
        "pages": 224,
        "description": "A humorous space adventure following Arthur Dent across the galaxy",
        "themes": "absurdity, technology, existence, humor",
        "setting": "Space, various planets"
    },
    {
        "id": "book_7",
        "title": "Dune",
        "author": "Frank Herbert",
        "genre": "Science Fiction",
        "year": 1965,
        "rating": 4.3,
        "pages": 688,
        "description": "A complex tale of politics, religion, and ecology on a desert planet",
        "themes": "power, ecology, religion, politics",
        "setting": "Arrakis, distant future"
    },
    {
        "id": "book_8",
        "title": "The Hunger Games",
        "author": "Suzanne Collins",
        "genre": "Dystopian",
        "year": 2008,
        "rating": 4.2,
        "pages": 374,
        "description": "A teenage girl fights for survival in a brutal televised competition",
        "themes": "survival, oppression, sacrifice, rebellion",
        "setting": "Panem, dystopian future"
    },]
     
    ids = [book["id"] for book in books]
    return books, ids

def get_books_documents(books):
    books_documents = []
    for book in books:
       document = f"""Book with title {book['title']} and author {book['author']} 
       and genre {book['genre']} and release at {book['year']}, having rating  of{book['rating']} 
       and pages {book['pages']} and description {book['description']} 
       and themes of the book is  {book['themes']} and settings {book['setting']}"""
       books_documents.append(document)
    return books_documents
def refine_book_filter(collection, query):
    result = collection.query(
        query_texts = [query],
        n_results = 2,
        where = {
            "$and": [
                {"rating": {"$gte": 4.0}},
                {"genre": {"$in": ["Fantasy", "Science Fiction"]}}
            ]
        }
    )
    return result

def book_collection(): 
    collection_name = "books_collection"
    books, ids = books_data()
    collection = create_collection(collection_name)
    collection.add(
        ids=ids,
        documents=get_books_documents(books),
        metadatas=[{
            "title": book["title"],
            "author": book["author"],
            "year": book["year"],   
            "rating": book["rating"],
            "pages": book["pages"],
            "description": book["description"],
            "themes": book["themes"],
            "setting": book["setting"],
            "genre": book["genre"],
        } for book in books]
    )
    return collection


if __name__ == "__main__":
    # collection = employee_collection()
    # perform_similarity_search(collection, "Python developer with web development experience")
    # perform_similarity_search(collection, "team leader manager with experience")
    # result = refine_filter(collection, "Python developer with web development experience")
    # for i, (doc_id, document, distance) in enumerate(zip(
    #     result['ids'][0], result['documents'][0], result['distances'][0]
    # )):
    #     metadata = result['metadatas'][0][i]
    #     print(f"  {i+1}. {metadata['name']} ({doc_id}) - Distance: {distance:.4f}")
    #     print(f"     {metadata['role']} in {metadata['location']} ({metadata['experience']} years)")
    #     print(f"     Document snippet: {document[:80]}...")
    collection = book_collection()
    # perform_similarity_search(collection, "magical fantasy adventure")
    result = refine_book_filter(collection, "magical fantasy adventure")
    for i, (doc_id, document, distance) in enumerate(zip(
        result['ids'][0], result['documents'][0], result['distances'][0]
    )):
        metadata = result['metadatas'][0][i]
        print(f"  {i+1}. {metadata['title']} ({doc_id}) - Distance: {distance:.4f}")
        print(f"     {metadata['author']} in {metadata['year']} years ({metadata['rating']} rating)")
        print(f"     Document snippet: {document[:80]}...")