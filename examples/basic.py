from bge_small_onnx import Encoder

encoder = Encoder.from_huggingface()
query = encoder.encode_queries(["small Australian marsupial"])
documents = encoder.encode_documents(
    [
        "A quokka is a small marsupial native to Western Australia.",
        "The service recovered after the database restart.",
    ]
)

print((query @ documents.T)[0].tolist())
