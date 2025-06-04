import os
import socket
import selectors
import traceback
from libserver import Message, wiki_model_from_path
import argparse

sel = selectors.DefaultSelector()


def accept_wrapper(sock):
    conn, addr = sock.accept()  # Should be ready to read
    #print("accepted connection from", addr)
    conn.setblocking(False)
    message = Message(sel, conn, addr)
    sel.register(conn, selectors.EVENT_READ, data=message)


def main():
    # Initialize the model with BERT embeddings
    wiki_model_from_path(None)  # Pass None to use BERT embeddings

    # host, port = sys.argv[1], int(sys.argv[2])
    host, port = "0.0.0.0", 9600
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Avoid bind() exception: OSError: [Errno 48] Address already in use
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind((host, port))
    lsock.listen()
    print("listening on", (host, port))

    lsock.setblocking(False)
    sel.register(lsock, selectors.EVENT_READ, data=None)

    try:
        while True:
            events = sel.select(timeout=None)
            for key, mask in events:
                if key.data is None:
                    accept_wrapper(key.fileobj)
                else:
                    message = key.data
                    try:
                        message.process_events(mask)
                    except Exception:
                        print(
                            "main: error: exception for",
                            f"{message.addr}:\n{traceback.format_exc()}",
                        )
                        message.close()
    except KeyboardInterrupt:
        print("caught keyboard interrupt, exiting")
    finally:
        sel.close()


if __name__ == "__main__":
    print("Starting word embedding server with BERT embeddings...")
    main()
