from fastapi import FastAPI
import uvicorn

from server.sentiment_v1.server_builder import ServerBuilder as SentimentV1ServerBuilder


class Server:
    TITLE = "Sentiment Analysis APP"
    VERSION = "1.0.0"

    CONTACT_NAME = "Sreram K"
    CONTACT_URL = "https://www.linkedin.com/in/k-sreram-a04a90b7/"
    CONTACT_EMAIL = "sreramk26@gmail.com"

    LICENSE_NAME = "Apache 2.0"

    LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0.html"

    def __init__(self, port=8080, host='0.0.0.0'):
        self.__port = port
        self.__host = host
        self.__app = Server.__initialize_server()

    @staticmethod
    def __initialize_server():
        return FastAPI(
            title=Server.TITLE,
            description=SentimentV1ServerBuilder.get_description(),
            version=Server.VERSION,
            contact={
                "name": Server.CONTACT_NAME,
                "url": Server.CONTACT_URL,
                "email": Server.CONTACT_EMAIL,
            },
            license_info={
                "name": Server.LICENSE_NAME,
                "url": Server.CONTACT_URL,
            },
            openapi_tags=SentimentV1ServerBuilder.get_tags_metadata(),
        )

    def build_endpoints(self):

        sentiment_v1_server_builder = SentimentV1ServerBuilder(self.__app)
        sentiment_v1_server_builder.build_server()

    def run_server(self):
        if self.__app is None:
            raise Exception("running the server before a call to build_endpoints is forbidden")

        uvicorn.run(self.__app, port=self.__port, host=self.__host)
