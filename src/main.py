import os
import time
import csv
from crewai import Crew
from langchain_groq import ChatGroq
from agents import EmailPersonalizationAgents
from tasks import PersonalizeEmailTask

# 0. Setup environment
from dotenv import load_dotenv
load_dotenv()

email_template = """
Dear [Name],

I hope this message finds you well.

I wanted to extend a warm invitation to join our Skool community. 
We host weekly coaching calls every Tuesday at 6 PM Eastern Time, and we'd love for you to be a part of it.
Our community is completely free, and we are excited to announce that we're about to reach a milestone of 500 users.

This community is a fantastic resource for getting support with your projects and connecting with 
others who share your interests. Whether you have questions or need assistance, 
our members and coaching calls are here to help.

We look forward to welcoming you to our community and seeing you at our next coaching call. 
If you have any questions or need further information, please don't hesitate to reach out.

Best regards,
Shivam Chhetry
"""

# 1. Create agents
agents = EmailPersonalizationAgents()

email_presonalizer = agents.personalize_email_agent()
ghostwriter = agents.ghostwriter_agent()


# 2. Create tasks
tasks = PersonalizeEmailTask()

personalize_email_tasks = []
ghostwriter_email_tasks = []

csv_file_path = "../data/clients_medium.csv"

# Open the CSV file
with open(csv_file_path, mode='r', newline='') as file:
  # Create a CSV reader object
  csv_reader = csv.DictReader(file)
  for row in csv_reader:
    # access each field in the row
    recipient = {
            'first_name': row['first_name'],
            'last_name': row['last_name'],
            'email': row['email'],
            'bio': row['bio'],
            'last_conversation': row['last_conversation']
    }

    # Create a personalize_email task for each recipient
    personalize_email_task = tasks.personalize_email(
       agent=email_presonalizer,
       recipient=recipient,
       email_template=email_template
    )

    # Create a ghostwriter_email task for each recipient
    ghostwriter_email_task = tasks.ghostwrite_email(
      agent=ghostwriter,
      draft_email=personalize_email_task,
      recipient=recipient
    )

    personalize_email_tasks.append(personalize_email_task)
    ghostwriter_email_tasks.append(ghostwriter_email_task)

# 3. Setup Crew

crew = Crew(
  agents=[email_presonalizer, ghostwriter],
  tasks = [
    *personalize_email_tasks,
    *ghostwriter_email_tasks
  ],
  max_rpm=29
)


# 4. Kick off the crew

start_time = time.time()

results = crew.kickoff()

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Crew kickoff took {elapsed_time} seconds.")
print("Crew usage", crew.usage_metrics)