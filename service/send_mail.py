import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import re
import os
from dotenv import load_dotenv
from email.mime.application import MIMEApplication


load_dotenv()

def is_valid_email(email):
    """Check if the provided email address is valid."""
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None

def send_email(subject, body, to_email, document_path=None):
    sender_email = os.getenv("MAIL")
    sender_password = os.getenv("APP_PASSWORD")

    # Banner image (hosted on Cloudinary)
    banner_url = "https://res.cloudinary.com/darid8ehu/image/upload/v1743309369/uwclev5dfhegnjsxjjbn.png"

    # Card-style email body with banner header
    formatted_body = f"""
    <div style="max-width: 600px; margin: auto; border-radius: 10px; overflow: hidden; 
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1); font-family: Arial, sans-serif; border: 1px solid #ddd;">
        
        <!-- Banner Header -->
        <div style="text-align: center;">
            <img src="{banner_url}" alt="Email Banner" style="width: 100%; height: auto; display: block;">
        </div>

        <!-- Email Card -->
        <div style="background: #ffffff; padding: 20px;">
            <h2 style="color: #2c3e50; text-align: center;">{subject}</h2>
            <p style="color: #333; font-size: 16px; line-height: 1.6;">
                {body.replace("\n", "<br>")}
            </p>
        </div>
    </div>
    """

    # Footer with social links
    footer = """
    <div style="background: linear-gradient(135deg, #e0f7fa, #b2ebf2); color: #34495e; padding: 30px; text-align: center; 
                border-top: 5px solid #e74c3c; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); margin-top: 20px;">
        <img src="https://res.cloudinary.com/darid8ehu/image/upload/v1743272982/hio0lln7gixfig5ykttk.jpg" alt="PersonaAI Icon" 
             style="width: 60px; height: 60px; vertical-align: middle; margin-bottom: 15px;">
        <div style="font-size: 28px; font-weight: bold; color: #e74c3c; margin: 10px 0;">PersonaAI</div>
        <div style="font-size: 16px; color: #7f8c8d; margin: 5px 0;">Developed by Anubhab Nandi</div>
        <div style="font-size: 14px; color: #34495e; margin: 10px 0;">A cutting-edge AI solution designed to enhance user interactions and automate tasks efficiently.</div>
        <div style="font-size: 14px; color: #34495e; margin: 10px 0;">For inquiries, contact: 
            <a href="mailto:anubhabnandi12@gmail.com" style="color: #e74c3c;">anubhabnandi12@gmail.com</a>
        </div>
        <div style="margin-top: 15px;">
            <a href="https://www.facebook.com/anubhab.nandi.73" target="_blank" style="text-decoration: none; color: #e74c3c; 
               margin: 0 15px; font-weight: bold;">
                <img src="https://img.icons8.com/color/48/000000/facebook.png" alt="Facebook" 
                     style="width: 24px; height: 24px; vertical-align: middle;"> Facebook
            </a>
            <a href="https://www.linkedin.com/in/anubhab-nandi-08a342215" target="_blank" style="text-decoration: none; 
               color: #e74c3c; margin: 0 15px; font-weight: bold;">
                <img src="https://img.icons8.com/color/48/000000/linkedin.png" alt="LinkedIn" 
                     style="width: 24px; height: 24px; vertical-align: middle;"> LinkedIn
            </a>
            <a href="https://github.com/anubhab12-bot" target="_blank" style="text-decoration: none; color: #e74c3c; 
               margin: 0 15px; font-weight: bold;">
                <img src="https://img.icons8.com/color/48/000000/github.png" alt="GitHub" 
                     style="width: 24px; height: 24px; vertical-align: middle;"> GitHub
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
    msg.attach(MIMEText(full_body, "html"))  # HTML version with styled card, banner, and footer

    if document_path and os.path.isfile(document_path):
        try:
            with open(document_path, 'rb') as doc_file:
                doc_attachment = MIMEApplication(doc_file.read(), _subtype='pdf')  # Change _subtype if not PDF
                doc_attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(document_path))
                msg.attach(doc_attachment)
        except Exception as e:
            print(f"Error attaching document: {str(e)}")

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

