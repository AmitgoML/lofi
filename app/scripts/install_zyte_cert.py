import certifi
from pathlib import Path

CERT_PATH = Path(__file__).parent.parent / "certs" / "zyte-ca.crt"


def install_cert():
    cert_data = CERT_PATH.read_text()

    cafile = certifi.where()
    print(f"Appending Zyte CA cert to {cafile}")

    with open(cafile, "r+") as f:
        contents = f.read()
        if cert_data not in contents:
            f.write("\n" + cert_data + "\n")
            print("Zyte CA certificate installed into certifi bundle.")
        else:
            print("Zyte CA certificate already present.")


if __name__ == "__main__":
    install_cert()
