import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import re
import os
from dotenv import load_dotenv

load_dotenv()

def is_valid_email(email):
    """Check if the provided email address is valid."""
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None

def send_email(subject, body, to_email):
    sender_email = os.getenv("MAIL")
    sender_password = os.getenv("APP_PASSWORD")

    formatted_body = "<p>" + body.replace("\n", "</p><p>") + "</p>"

    footer = """
    <div style="background: linear-gradient(135deg, #e0f7fa, #b2ebf2); color: #34495e; padding: 30px; text-align: center; border-top: 5px solid #e74c3c; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);">
        <img src="https://res.cloudinary.com/darid8ehu/image/upload/v1743272982/hio0lln7gixfig5ykttk.jpg" alt="PersonaAI Icon" style="width: 60px; height: 60px; vertical-align: middle; margin-bottom: 15px;">
        <div style="font-size: 28px; font-weight: bold; color: #e74c3c; margin: 10px 0;">PersonaAI</div>
        <div style="font-size: 16px; color: #7f8c8d; margin: 5px 0;">Developed by Anubhab Nandi</div>
        <div style="font-size: 14px; color: #34495e; margin: 10px 0;">A cutting-edge AI solution designed to enhance user interactions and automate tasks efficiently.</div>
        <div style="font-size: 14px; color: #34495e; margin: 10px 0;">For inquiries, contact: <a href="mailto:anubhabnandi12@gmail.com" style="color: #e74c3c;">anubhabnandi12@gmail.com</a></div>
        <div style="margin-top: 15px;">
            <a href="https://www.facebook.com/anubhab.nandi.73" target="_blank" style="text-decoration: none; color: #e74c3c; margin: 0 15px; font-weight: bold;">
                <img src="https://img.icons8.com/color/48/000000/facebook.png" alt="Facebook" style="width: 24px; height: 24px; vertical-align: middle;"> Facebook
            </a>
            <a href="https://www.linkedin.com/in/anubhab-nandi-08a342215" target="_blank" style="text-decoration: none; color: #e74c3c; margin: 0 15px; font-weight: bold;">
                <img src="https://img.icons8.com/color/48/000000/linkedin.png" alt="LinkedIn" style="width: 24px; height: 24px; vertical-align: middle;"> LinkedIn
            </a>
            <a href="https://github.com/anubhab12-bot" target="_blank" style="text-decoration: none; color: #e74c3c; margin: 0 15px; font-weight: bold;">
                <img src="https://img.icons8.com/color/48/000000/github.png" alt="GitHub" style="width: 24px; height: 24px; vertical-align: middle;"> GitHub
            </a>
        </div>
    </div>
    """
    
    full_body = f"{formatted_body}<br><br>{footer}"

    # Create a multipart message
    msg = MIMEMultipart("alternative")
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject

    # Attach both plain text and HTML versions
    msg.attach(MIMEText(body, "plain"))  # Plain text version
    msg.attach(MIMEText(full_body, "html"))  # HTML version with footer

    try:
        # Connect to Gmail SMTP Server
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()  # Secure the connection
        server.login(sender_email, sender_password)  # Login to the email account
        server.sendmail(sender_email, to_email, msg.as_string())  # Send the email
        server.quit()  # Close the connection
        print("Email sent successfully!")
    except Exception as e:
        print("Error sending email:", e)
