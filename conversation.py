import argparse
import enum
import os
from datetime import datetime

from google import generativeai as gemini
import openai

class AiAgentType(enum.Enum):
    """Enum for supported AI agent types."""
    GEMINI = enum.auto()
    OPENAI = enum.auto()

class AiAgent:
    """Base class for AI agents."""

    def __init__(self, agent_role: str) -> None:
        del agent_role
        raise NotImplementedError()
    
    def generate_response(self, input_text: str) -> str:
        del input_text
        raise NotImplementedError()

class GeminiAgent(AiAgent):
    """Agent implementation using Google's Gemini API."""

    def __init__(self, agent_role: str) -> None:
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        gemini.configure(api_key=gemini_api_key)
        self.model = gemini.GenerativeModel('gemini-1.5-flash-002', system_instruction=agent_role)
        self.chat = self.model.start_chat(history=[])
    
    def generate_response(self, input_text: str) -> str:
        response = self.chat.send_message(input_text)
        return response.text    

class OpenAiAgent(AiAgent):
    """Agent implementation using OpenAI's API."""

    def __init__(self, agent_role: str) -> None:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        openai.api_key = openai_api_key
        self.client = openai.OpenAI()
        self.messages = [{"role": "system", "content": agent_role}]
    
    def generate_response(self, input_text: str) -> str:
        self.messages.append({"role": "user", "content": input_text})
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages
        )
        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})
        return assistant_message

class Interview:
    """Manages an AI-driven interview session."""

    def __init__(self, agent_type: AiAgentType, num_turns: int, 
                 persona_prompt: str, idea: str) -> None:
        self.num_turns = num_turns      

        agent_classes = {
            AiAgentType.GEMINI: GeminiAgent,
            AiAgentType.OPENAI: OpenAiAgent
        }
        try:
            agent_class = agent_classes[agent_type]
        except KeyError:
            raise ValueError(f"Invalid agent type: {agent_type}")
        
        # Generate a persona for the interviewee
        persona_generator_role = (
            "Design a persona for a simulation. Give a detailed description of: "
            "their role, the company they work for, the industry the company works for, "
            "total number of employees in the company, where the company is based, "
            "the products or services offered by the company, their background, how they got this job, "
            "their age, their interests, motivations, top of mind issues for you at your job, "
            "and other key employees within the company."
        )
       
        persona_generator = agent_class(persona_generator_role)
        self.interviewee_persona = persona_generator.generate_response(persona_prompt)
        print(f"Interviewee persona:\n{self.interviewee_persona}")

        # Initialize interviewee
        interviewee_instructions = (
            "Someone has reached out with a cold email asking if they can interview you because they want to ask a few "
            "questions about your job. You don't know this person, but you want to help so you will be very honest and "
            "thoughtful in your responses to their questions. You should answer according to your persona and the type of "
            "company you work for. Sometimes they might ask about the challenges of doing certain things, then think about "
            "whether that's challeging or not for your persona, and if it is, think of the reasons why. Ask clarifying "
            "questions if needed. Behave naturally, as you would in a real conversation (remember this is a dialog, so "
            "don't use bullet points or titles). Remember this person is looking for very honest feedback, so don't try to "
            "please them of be overly optimistic - just provide honest answers and share your insights. When asked for "
            "your thoughts about potential solutions or ideas, don't be afraid of being critical, since they will appreciate "
            "constructive feedback. If you think the solutions discussed would be useful to your persona, you can say that and also "
            "build on the idea. If you come up with ideas during the conversation you can also say that. Don't ask "
            "the interviewer for their opinion, they aren't an expert on this area and they are only trying to learn from you."
        )
        interviewee_role = (
            f"{interviewee_instructions}\n\n"
            f"This is your persona - at all times you should answer as if you were this person:\n\n"
            f"{self.interviewee_persona}"
        )
        self.interviewee = agent_class(interviewee_role)

        # Initialize interviewer
        interviewer_prompt = (
            f"You're an entrepreneur thinking of starting a new company. You have some ideas related to {idea}. However, "
            "first you need to understand the problem you're trying to solve and understand your customers' pain points. For "
            "that reason, you're conducting customer interviews. You have a beginner's mindset - stay courious and always try "
            "to learn from the interviewee. The intervieww know much more than you about this and you're trying to learn. Follow "
            "the guidelines from the book \"The Mom Test\" and make sure you aren't guiding the witness - you won't disclose your idea "
            "to the customer, but instead just try to uncover valuable insights about what their job is like and what are the "
            "problems that are strong enough such that they might be willing to pay for a solution. You want to learn the "
            "user's needs and frustrations related to the problem you're trying to solve. You should ask open-ended questions "
            "to gather information about the interviewee's experiences, needs, and opinions related to the problem your product "
            "solves. You will receive a response for every question you ask, but you should only ask one question at a time. "
            "Be brief, don't talk to much or give advice / suggestions to the interviewee, instead focus on asking questions "
            "you need to learn as much as possible from them. Listen to the interviewee responses and follow interesting "
            "threads that you think might be relevant to your research. Try to go really deep: understand why hard things are "
            "hard, dig into what their day to day pain points are. You want to uncover non-obvious insights. If they say "
            "something is challenging or a pain point for them, discuss solutions they have tried and try to see if those are "
            "sufficient for them. If they have clarifying questions, answer providing more detail, but remember to keep "
            "questions open ended. This is a long interview in which you will ask a total of 20 questions, so keep the "
            "discussion going. When you have a good understanding of the problem you can discuss how potential AI-based "
            "solutions would help this person in their workflow, but only do this when you have exhausted your questions about "
            "their challenges and pain points. Include \"[END OF INTERVIEW]\" in your response when the dialog has ended."
        )
        self.interviewer = agent_class(interviewer_prompt)

        # Initialize conversation history
        self.conversation = []
    

    def conduct_interview(self) -> None:
        
        # Generate initial question
        initial_question = self.interviewer.generate_response("Start the interview by introducing yourself and the purpose of the interview.")
        self.conversation.append(("Interviewer", initial_question))
        print("Interviewer:", initial_question)

        for _ in range(self.num_turns):
            # Get interviewee's response
            last_question = self.conversation[-1][1]
            interviewee_response = self.interviewee.generate_response(last_question)
            self.conversation.append(("Interviewee", interviewee_response))
            print("Interviewee:", interviewee_response)
            
            # Get interviewer's next question
            interviewer_question = self.interviewer.generate_response(interviewee_response)
            self.conversation.append(("Interviewer", interviewer_question))
            print("Interviewer:", interviewer_question)
            
            # Check if the interviewer has ended the interview
            if "[END OF INTERVIEW]" in interviewer_question:
                print("The interviewer has ended the interview.")
                break

    def save_conversation(self) -> str:
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/interview_{timestamp}.txt"
        
        # Write conversation to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Intervieweee persona: {self.interviewee_persona}\n\n\nStart of interview\n\n")

            for role, text in self.conversation:
                f.write(f"{role}: {text}\n\n")
        
        return filename

    def save_interviewee_responses(self) -> str:
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/interviewee_responses_{timestamp}.txt"
        
        # Write conversation to file
        with open(filename, 'w', encoding='utf-8') as f:
            for role, text in self.conversation:
                if role == 'Interviewee':
                    f.write(f"{text}\n\n")
        
        return filename


def main() -> None:
    parser = argparse.ArgumentParser(description='Conduct AI interviews')
    parser.add_argument('--persona', type=str, required=True,
                      help='Prompt to generate the interviewee persona')
    parser.add_argument('--idea', type=str, required=True,
                      help='The business idea to explore')
    parser.add_argument('--repetitions', type=int, default=15,
                      help='Number of interviews to conduct (default: 15)')
    parser.add_argument('--turns', type=int, default=20,
                      help='Max number of turns in an interview (default: 20)')
    parser.add_argument('--model', type=str, default='openai',
                    help='Can be openai or gemini (default: openai).')

    args = parser.parse_args()

    for _ in range(args.repetitions):

        _ai_models_args = {
            'gemini': AiAgentType.GEMINI,
            'openai': AiAgentType.OPENAI
        }

        if args.model not in _ai_models_args:
            print(f"Error: Model '{args.model}' not recognized. Must be one of: {', '.join(_ai_models_args.keys())}")
            return

        # Create and conduct interview
        interview = Interview(_ai_models_args[args.model], num_turns=args.turns, 
                            persona_prompt=args.persona, idea=args.idea)
        interview.conduct_interview()
        
        # Save the conversation
        filename = interview.save_conversation()
        print(f"Interview saved to {filename}")

        # Save only the interviewee responses to a separate file
        filename = interview.save_interviewee_responses()
        print(f"Responses saved to {filename}")

if __name__ == "__main__":
    main()
