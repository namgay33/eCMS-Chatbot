from transformers import pipeline

# Initialize the summarization pipeline with the specified model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Input text to be summarized
text = """
eCMS will allow unregistered Companies and users to apply for registration through an application process 
through a link provided in the eCMS login page. After submission of application, it shall be reviewed by Customs. 
Based on review outcomes if the application is approved, then, it shall be registered in eCMS. 
1: Visit the Registration Page: Go to the eCMS Bhutan registration page. 
2: OTP Authentication: Enter your TPN, mobile number, or email ID to generate an OTP. 
3: Validate OTP: Enter the received OTP to validate. 
4: Select Registration Type: Choose either Individual or Organization registration. 
5: Fill Required Details: Provide all the necessary details as prompted. 
6: Required Details: 
I: Individual: TPN number and CID. 
ii: Business: TPN number and valid trade license. 
iii: Company: TPN number and registration number.
"""

# Calculate text length (number of characters in the input text)
text_length = len(text)

# Set max_length to 3/4 of the text length or a reasonable upper limit
max_length = min(int(text_length * 0.75), 300) 

# Set min_length to be at least 30 characters or 1/10 of the text length
min_length = max(int(text_length * 0.75), 30)

# Generate the summary with dynamically set lengths
summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)

# Print the summarized text
print(summary[0]['summary_text'])
