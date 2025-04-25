from SkipGram import WebSessionEmbedder

if __name__ == '__main__':
    embedder = WebSessionEmbedder(
        use_fields=['method', 'decoded_path', 'status']
    )
    embedder.load_model("models")

    request = {
        "timestamp": "2019-01-22T14:29:33+03:30",
        "method": "GET",
        "decoded_path": "/image/332/mainSlide",
        "status": 200,
        "size": 95987
    }

    embed = embedder.get_request_embedding(request)
    print(embed)