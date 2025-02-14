# Update run_with_ngrok.py with this code
from pyngrok import ngrok
import os

# Set your authtoken (replace YOUR_AUTH_TOKEN with your actual token)
ngrok.set_auth_token("2t1TnY16BbLoWt7Du0J3qqDDcAZ_2fcQ3CyVhFuojHnAm8g2")

# Start ngrok
public_url = ngrok.connect(port=8501)
print(f' * Public URL: {public_url}')

# Run streamlit
os.system('streamlit run app.py')

