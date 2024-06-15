import os
from pymilvus import MilvusClient
from PIL import Image
from feature_store import FeatureExtractor
from IPython.display import display
from pymilvus import connections, db
from milvus import default_server


def create():
    # client = MilvusClient(
    #     uri="http://127.0.0.1:19530"
    # )
    #
    # # 2. Create a collection
    # client.create_collection(
    #     collection_name="quick_setup",
    #     dimension=5,
    #     metric_type="IP"
    # )
    # client = MilvusClient("http://localhost:19530")
    #
    client = MilvusClient(
        uri="http://root:Milvus@localhost:19530",
        db_name="default"
    )

    client.create_collection(
        collection_name="image_embeddings",
        vector_field_name="vector",
        dimension=512,
        auto_id=True,
        enable_dynamic_field=True,
        metric_type="L2",
    )


    # Set up a Milvus client
    # client = MilvusClient("dogs.db")
    # Create a collection in quick setup mode
    # client.create_collection(
    #     collection_name="image_embeddings",
    #     vector_field_name="vector",
    #     dimension=512,
    #     auto_id=True,
    #     enable_dynamic_field=True,
    #     metric_type="COSINE",
    # )
    return client


def insert(client):
    extractor = FeatureExtractor("resnet34")
    root = "./temp_data/train"
    insert = True
    if insert is True:
        for dirpath, foldername, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith(".JPEG"):
                    filepath = dirpath + "/" + filename
                    image_embedding = extractor(filepath)
                    client.insert(
                        "image_embeddings",
                        {"vector": image_embedding, "filename": filepath},
                    )


def compare():
    # query_image = "./test/Afghan_hound/n02088094_4261.JPEG"
    query_image = "./temp_data/test/Airedale/n02096051_4092.JPEG"

    extractor = FeatureExtractor("resnet34")

    results = client.search(
        "image_embeddings",
        data=[extractor(query_image)],
        output_fields=["filename"],
        search_params={"metric_type": "L2"},
        # search_params={"metric_type": "COSINE"},
    )
    print(results)
    images = []
    for result in results:
        for hit in result[:10]:
            filename = hit["entity"]["filename"]
            img = Image.open(filename)
            img = img.resize((150, 150))
            images.append(img)

    width = 150 * 5
    height = 150 * 2
    concatenated_image = Image.new("RGB", (width, height))

    for idx, img in enumerate(images):
        x = idx % 5
        y = idx // 5
        concatenated_image.paste(img, (x * 150, y * 150))
    display("query")
    display(Image.open(query_image).resize((150, 150)))
    display("results")
    display(concatenated_image)


if __name__ == '__main__':
    client = create()
    print("Collection has been created!")
    # insert(client)
    print("Data has been inserted!")
    compare()
