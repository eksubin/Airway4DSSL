# start the mlflow tracking
from pyngrok import ngrok
get_ipython().system_raw("mlflow ui --port 5000 &")


ngrok.kill()

# Replace with your ngrok auth token
ngrok.set_auth_token("2K0Su18e3RupXt2nhqBPds10VN0_5nVVwNWUAQBa8ZCVvgQ9c")

# Start ngrok tunnel
public_url = ngrok.connect(addr="5000", proto="http", bind_tls=True).public_url
print("ngrok tunnel URL:", public_url)
