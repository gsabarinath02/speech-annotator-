from __future__ import annotations

EXAMPLE_SCRIPTS: list[dict[str, str]] = [
    {
        "title": "Mayo Meds to Beds - Declined Delivery",
        "text": """[navigator] Hello John Doe. This is the Mayo Clinic Pharmacy calling on a recorded line about our Meds to Beds delivery service. Is now a good time to talk?
[user] It is.
[navigator] Thank you. To make sure I reached the right room, am I speaking with John Doe?
[user] Yes. You are.
[navigator] Great, thank you. For privacy, can you please state your date of birth, including the year?
[user] This twelve of April nineteen fifty eight.
[navigator] Just to confirm, that birthdate is April twelfth, nineteen fifty-eight?
[user] Yes.
[navigator] Thank you, John. Your care team has prescribed medications for you to take at home. For your convenience, the Mayo Clinic Pharmacy can fill your prescriptions and deliver them right to your bedside through our Meds to Beds program, which is a free service we offer at Mayo Clinic. Would you like Mayo Clinic Pharmacy to fill these through Meds to Beds and deliver them to your room? Participation is optional.
[user] No. I'm not interested.
[navigator] Thank you for letting me know. No delivery will be made. I will transfer you to a pharmacy team member for any further assistance. Please hold on a moment.""",
    },
    {
        "title": "Virtual Care - Neutral",
        "text": """[neutral] Hello, this is Maya from the Outcomes care team.
[verification] For safety, please confirm your full date of birth, including the year.
[neutral] I have July 18, 1964 on file.
[acknowledging] Thank you.
[instruction] Today I will help you complete your remote patient monitoring setup.
[instruction] Please place the tablet, blood pressure monitor, cuff, pulse oximeter, and scale in front of you.
[instruction] First, turn on the tablet and enter the username and password from your welcome packet.
[instruction] Once you are signed in, choose your preferred language and tap Save.
[instruction] If you see the Terms of Service, please scroll to the bottom and tap Accept.
[instruction] Next, sit in a chair with your back supported and your feet flat on the floor.
[instruction] Place the cuff on your upper arm with the tubing pointing downward and keep your arm relaxed.
[instruction] Tap Take Blood Pressure and save the reading when it appears.
[instruction] Then place the pulse oximeter on one finger and keep your hand still until the numbers settle.
[instruction] Save that reading on the tablet.
[instruction] Finally, place the scale on a hard surface, step on carefully, and wait for the number to settle before saving.
[instruction] After your readings are complete, you can use the chat icon in the top corner of the tablet if you need non-urgent help from the care team.
[close] Your setup is complete, and your care team will now be able to review your readings.
[instruction] If you ever have questions or need help, you can always call the care team at 507-293-3371. If anything comes up, don't hesitate to reach out.
[close] Alright! You've done a fantastic job today, and your setup is complete. Is there anything else I can help you with before we end the call?""",
    },
    {
        "title": "Virtual Care - Warm and Reassuring",
        "text": """[neutral] Hello, this is Sarah from the Outcomes care team.
[warm] I'm glad we could connect today.
[verification] For privacy, please confirm your full date of birth, including the year.
[neutral] Thank you for clarifying, Linda. Just to confirm, your last name is Anderson, spelled A N D E R S O N, correct? And your date of birth is May twenty fifth, nineteen sixty four?
[acknowledging] Thank you.
[empathetic] I understand you are ready to return your monitoring equipment, and I'll walk through it step by step with you.
[instruction] Please place the tablet, blood pressure monitor, cuff, pulse oximeter, and scale near the shipping box.
[instruction] Before sealing the box, please make sure each device is switched off.
[instruction] Place the blood pressure monitor and cuff in first, then the pulse oximeter, the scale, the tablet, and the paperwork if you still have it.
[instruction] If you no longer have the original box, any sturdy box is okay as long as everything fits securely.
[instruction] Please attach the return shipping label on the top and remove or cover any older shipping labels.
[verification] I want to confirm the pickup address as 26 Meadow Lane, Unit 12, Phoenix, Arizona, 85014.
[acknowledging] Thank you.
[instruction] I have the UPS pickup scheduled for Tuesday, January 6, between 9 AM and 12 PM.
[instruction] Please place the sealed box near the door or in a spot that is easy for the driver to access.
[warm] If anything changes, your care team can help update the pickup.
[close] Thank you for your participation in the program, and please don't hesitate to reach out if you need anything else.""",
    },
    {
        "title": "Virtual Care - Urgent but Calm",
        "text": """[neutral] Hello, this is Daniel from the Outcomes care team.
[verification] For safety, please confirm your full date of birth, including the year.
[neutral] I have November 22, 1974 on file.
[acknowledging] Thank you.
[calm] I'm calling because we have not received your vital readings for the past two days, and I'd like to help you get back on track.
[de-escalating] Sometimes the reason is simple, like a busy schedule, low batteries, trouble opening the app, or not feeling well enough to take the readings.
[warm] If the equipment has been difficult to use, that is completely okay, and we can work through it together.
[instruction] Please place the tablet and blood pressure monitor in front of you.
[instruction] Open the app and look for the button that says Take Blood Pressure under Your Tasks.
[instruction] Sit with your back supported, your feet flat on the floor, and your arm relaxed.
[instruction] Start the reading and save it once it appears.
[instruction] Then place the pulse oximeter on one finger and save that reading when the numbers become steady.
[calm] If you missed readings because of dizziness, weakness, shortness of breath, or any other new symptom, please let your care team know.
[de-escalating] If you feel frustrated with the equipment or have had repeated trouble, I can note that so someone can follow up and help.
[warm] These daily readings help your care team notice changes early, so getting back into a routine can make a real difference.
[close] Once today's readings are complete, you should be back on track.""",
    },
    {
        "title": "Triage - Neutral",
        "text": """[neutral] Thank you for calling the nurse triage line.
[neutral] For quality and training purposes, this call may be recorded.
[calm] If this is a medical emergency, please hang up and call 911 now.
[verification] Please confirm your full name and your date of birth, including the year.
[neutral] I have August 14, 1968 on file.
[acknowledging] Thank you.
[neutral] I understand you are calling about abdominal pain today.
[neutral] Please tell me when the pain started and whether it began suddenly or built up gradually over time.
[neutral] Tell me exactly where the pain is located.
[neutral] On a scale from zero to ten, how severe is it right now?
[neutral] Does the pain feel sharp, cramping, burning, pressure-like, or aching?
[neutral] Have you also had nausea, vomiting, fever, diarrhea, bloating, or trouble keeping fluids down?
[neutral] Have you noticed black stool, blood in stool, or vomit that looks dark like coffee grounds?
[neutral] Is the pain so strong that it is hard for you to stand up straight or do normal activities?
[neutral] Have you had this kind of pain before?
[neutral] Do you have any history of gallbladder problems, ulcers, bowel disease, or recent stomach infection?
[neutral] Have you started any new medicine recently, or have you tried anything so far, such as rest, fluids, or over-the-counter medication?
[close] Thank you. That gives the nurse a clearer picture of what is going on.""",
    },
    {
        "title": "Triage - Warm and Reassuring",
        "text": """[neutral] Thank you for calling the triage line.
[empathetic] I'm sorry you're not feeling well today.
[warm] I'll keep this simple and go step by step so the nurse has a clear picture of what is happening.
[calm] If this is a medical emergency, please hang up and call 911 right away.
[verification] First, please confirm your full name and date of birth, including the year.
[neutral] I have May 9, 1981 on file.
[acknowledging] Thank you.
[warm] In your own words, please tell me what symptoms are bothering you most right now.
[warm] Take your time.
[empathetic] I know it can be hard to explain when you are not feeling well.
[neutral] Are you having fever, chills, sore throat, cough, body aches, headache, fatigue, nausea, or dizziness?
[neutral] When did these symptoms start, and which symptom came first?
[neutral] Are they getting worse, staying the same, or starting to improve?
[neutral] Are you able to drink fluids and keep them down?
[neutral] Have you been able to stand and walk normally today?
[neutral] Have you checked your temperature, and if so, what was the highest reading?
[neutral] Are you having any chest discomfort, shortness of breath, or wheezing?
[neutral] Have you been around anyone who was sick recently, or have you had a recent flu or COVID exposure?
[neutral] Do you have asthma, diabetes, heart disease, or any other ongoing medical condition?
[neutral] What medicines have you taken today to try to feel better?
[close] Thank you. These details will help the nurse decide the safest next step.""",
    },
    {
        "title": "Triage - Urgent but Calm",
        "text": """[neutral] Thank you for calling the triage line.
[calm] I need to ask a few quick safety questions.
[calm] If you are having severe trouble breathing, chest pain that is getting worse, fainting, blue lips, sudden confusion, or symptoms of stroke, call 911 immediately.
[verification] Please confirm your full name and date of birth.
[neutral] I have January 27, 1956 on file.
[acknowledging] Thank you.
[instruction] If we get disconnected during this urgent review, I have your callback number as 303-555-4419. If there is a better number, please let me know.
[urgent but calm] Please stay near your phone.
[calm] I understand this began after a new medication, and I want to check for signs of a serious reaction.
[neutral] Are you having hives, a spreading rash, swelling of the lips or tongue, wheezing, trouble swallowing, chest tightness, dizziness, or feeling like you may pass out?
[neutral] Please tell me the name of the medication, the dose, and the time you last took it.
[neutral] How long after taking it did the symptoms begin?
[neutral] Have you taken this medication before, or is this the first dose?
[neutral] Have you taken anything else today, such as another prescription, over-the-counter medicine, or supplement?
[calm] Because swelling or breathing symptoms after a new medication can become serious quickly, I am documenting this as urgent.
[instruction] Please do not take another dose unless a clinician tells you to.
[calm] Please stay near your phone and remain available for immediate nurse follow-up or transfer.""",
    },
]

PROMPTS: list[str] = [script["text"] for script in EXAMPLE_SCRIPTS]
PROMPTS_BY_SUFFIX = {f"{index:02d}": prompt.upper() for index, prompt in enumerate(PROMPTS)}
