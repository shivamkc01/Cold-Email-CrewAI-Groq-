import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from crewai import Agent

load_dotenv()

class EmailPersonalizationAgents():
  def __init__(self):
    self.llm =  ChatGroq(
      api_key=os.getenv("GROQ_API_KEY"),
      model="mixtral-8x7b-32768"
    )

  def personalize_email_agent(self):
    return Agent(
      role="Email Personalizer",
      goal = f"""
        Personalize template emails for recipients using their information.

        GIven a template email and recipient information (name, email, bio, last conversation),
        personalize the email by incorporating the recipient's details
        into the email while maintaining the core message and structure of the original email.
        This involves updating the introduction, boy, and closing of the email to make 
        it more personal and engaging for each recipient.
        """,
      backstory="""
        As a Email Personalizer, you are responsible for customizing template emails for individual recipent
        based on their information and previous interactions.      
        """,
      verbose=True,
      llm=self.llm,
      max_iter=2
    ) 

  def ghostwriter_agent(self):
    return Agent(
            role="Ghostwriter",
            goal=f"""
                Revise draft emails to adopt the Ghostwriter's writing style.

                Use an informal, engaging, and slightly sales-oriented tone, mirroring the Ghostwriter's final email communication style.
                """,
            backstory="""
                As a Ghostwriter, you are responsible for revising draft emails to match the Ghostwriter's writing style, focusing on clear, direct communication with a friendly and approachable tone.
                """,
            verbose=True,
            llm=self.llm,
            max_iter=2,
    )