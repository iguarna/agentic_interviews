usage: conversation.py [-h] --persona PERSONA --idea IDEA [--repetitions REPETITIONS] [--turns TURNS] [--model MODEL]

Conduct AI interviews

options:

  -h, --help            show this help message and exit
  
  --persona PERSONA     Prompt to generate the interviewee persona
  
  --idea IDEA           The business idea to explore
  
  --repetitions REPETITIONS
                        Number of interviews to conduct (default: 15)
                        
  --turns TURNS         Max number of turns in an interview (default: 20)
  
  --model MODEL         Can be openai or gemini (default: openai).
