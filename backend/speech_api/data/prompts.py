from __future__ import annotations

EXAMPLE_SCRIPTS: list[dict[str, str]] = [
    {
        "title": "Mayo Meds to Beds - Declined Delivery",
        "text": """[neutral] [Navigator] Hello, John Doe. This is the Mayo Clinic Pharmacy calling on a recorded line about our Meds to Beds delivery service. Is now a good time to talk?
[neutral] [User] Yes, it is.
[verification] [Navigator] Thank you. To make sure I reached the right room, am I speaking with John Doe?
[neutral] [User] Yes, you are.
[verification] [Navigator] Great, thank you. For privacy, can you please state your date of birth, including the year?
[neutral] [User] It is April twelfth, nineteen fifty-eight.
[acknowledging] [Navigator] Just to confirm, that birthdate is April twelfth, nineteen fifty-eight?
[neutral] [User] Yes.
[acknowledging] [Navigator] Thank you, John.
[neutral] [Navigator] Your care team has prescribed medications for you to take at home.
[warm] [Navigator] For your convenience, the Mayo Clinic Pharmacy can fill your prescriptions and deliver them right to your bedside through our Meds to Beds program.
[neutral] [Navigator] This is a free service we offer at Mayo Clinic, and participation is optional.
[instruction] [Navigator] Would you like Mayo Clinic Pharmacy to fill these medications through Meds to Beds and deliver them to your room?
[neutral] [User] No, I'm not interested.
[acknowledging] [Navigator] Thank you for letting me know.
[neutral] [Navigator] No delivery will be made.
[instruction] [Navigator] I will transfer you to a pharmacy team member for any further assistance.
[close] [Navigator] Please hold on for a moment.""",
    },
    {
        "title": "Virtual Care - Neutral",
        "text": """[neutral] [Navigator] Hello, this is Maya from the Outcomes care team.
[neutral] [User] Hello.
[verification] [Navigator] For safety, please confirm your full date of birth, including the year.
[neutral] [User] July eighteenth, nineteen sixty-four.
[neutral] [Navigator] I have July 18, 1964 on file.
[acknowledging] [Navigator] Thank you.
[neutral] [User] Okay.
[instruction] [Navigator] Today I will help you complete your remote patient monitoring setup.
[neutral] [User] Alright.
[instruction] [Navigator] Please place the tablet, blood pressure monitor, cuff, pulse oximeter, and scale in front of you.
[neutral] [User] I have them all here.
[instruction] [Navigator] First, turn on the tablet and enter the username and password from your welcome packet.
[neutral] [User] Okay, give me a moment.
[neutral] [User] Alright, I'm logged in now.
[instruction] [Navigator] Once you are signed in, choose your preferred language and tap Save.
[neutral] [User] Done.
[instruction] [Navigator] If you see the Terms of Service, please scroll to the bottom and tap Accept.
[neutral] [User] Okay, I accepted it.
[instruction] [Navigator] Next, sit in a chair with your back supported and your feet flat on the floor.
[neutral] [User] I'm seated now.
[instruction] [Navigator] Place the cuff on your upper arm with the tubing pointing downward and keep your arm relaxed.
[neutral] [User] Alright, it's on.
[instruction] [Navigator] Tap Take Blood Pressure and save the reading when it appears.
[neutral] [User] Okay, it's taking the reading now.
[neutral] [User] I saved it.
[instruction] [Navigator] Then place the pulse oximeter on one finger and keep your hand still until the numbers settle.
[neutral] [User] Alright.
[neutral] [User] The numbers are steady now.
[instruction] [Navigator] Save that reading on the tablet.
[neutral] [User] Done.
[instruction] [Navigator] Finally, place the scale on a hard surface, step on carefully, and wait for the number to settle before saving.
[neutral] [User] Okay, one moment.
[neutral] [User] Alright, I've saved that too.
[instruction] [Navigator] After your readings are complete, you can use the chat icon in the top corner of the tablet if you need non-urgent help from the care team.
[neutral] [User] Okay, that's good to know.
[close] [Navigator] Your setup is complete, and your care team will now be able to review your readings.
[neutral] [User] Great.
[instruction] [Navigator] If you ever have questions or need help, you can always call the care team at 507-293-3371. If anything comes up, don't hesitate to reach out.
[neutral] [User] Alright, thank you.
[close] [Navigator] Alright! You've done a fantastic job today, and your setup is complete. Is there anything else I can help you with before we end the call?
[neutral] [User] No, that's all for now. Thank you.""",
    },
    {
        "title": "Virtual Care - Warm and Reassuring",
        "text": """[neutral] [Navigator] Hello, this is Sarah from the Outcomes care team.
[neutral] [User] Hi.
[warm] [Navigator] I'm glad we could connect today.
[neutral] [User] Me too.
[verification] [Navigator] For privacy, please confirm your full date of birth, including the year.
[neutral] [User] May twenty-fifth, nineteen sixty-four.
[neutral] [Navigator] Thank you for clarifying, Linda. Just to confirm, your last name is Anderson, spelled A N D E R S O N, correct? And your date of birth is May twenty-fifth, nineteen sixty-four?
[neutral] [User] Yes, that's correct.
[acknowledging] [Navigator] Thank you.
[neutral] [User] Okay.
[empathetic] [Navigator] I understand you are ready to return your monitoring equipment, and I'll walk through it step by step with you.
[neutral] [User] Thank you, I appreciate that.
[instruction] [Navigator] Please place the tablet, blood pressure monitor, cuff, pulse oximeter, and scale near the shipping box.
[neutral] [User] Alright, I have everything together.
[instruction] [Navigator] Before sealing the box, please make sure each device is switched off.
[neutral] [User] Okay, I'll check them now.
[neutral] [User] They're all off.
[instruction] [Navigator] Place the blood pressure monitor and cuff in first, then the pulse oximeter, the scale, the tablet, and the paperwork if you still have it.
[neutral] [User] Alright, I'm packing them now.
[instruction] [Navigator] If you no longer have the original box, any sturdy box is okay as long as everything fits securely.
[neutral] [User] I still have the original box.
[instruction] [Navigator] Please attach the return shipping label on the top and remove or cover any older shipping labels.
[neutral] [User] Okay, I've attached the label.
[verification] [Navigator] I want to confirm the pickup address as 26 Meadow Lane, Unit 12, Phoenix, Arizona, 85014.
[neutral] [User] Yes, that's the right address.
[acknowledging] [Navigator] Thank you.
[neutral] [User] Sure.
[instruction] [Navigator] I have the UPS pickup scheduled for Tuesday, January 6, between 9 AM and 12 PM.
[neutral] [User] Okay, that works.
[instruction] [Navigator] Please place the sealed box near the door or in a spot that is easy for the driver to access.
[neutral] [User] I can do that.
[warm] [Navigator] If anything changes, your care team can help update the pickup.
[neutral] [User] Good to know.
[close] [Navigator] Thank you for your participation in the program, and please don't hesitate to reach out if you need anything else.
[neutral] [User] Thank you for your help.""",
    },
    {
        "title": "Virtual Care - Urgent but Calm",
        "text": """[neutral] [Navigator] Hello, this is Daniel from the Outcomes care team.
[neutral] [User] Hello.
[verification] [Navigator] For safety, please confirm your full date of birth, including the year.
[neutral] [User] November twenty-second, nineteen seventy-four.
[neutral] [Navigator] I have November 22, 1974 on file.
[acknowledging] [Navigator] Thank you.
[neutral] [User] Yes.
[calm] [Navigator] I'm calling because we have not received your vital readings for the past two days, and I'd like to help you get back on track.
[neutral] [User] Oh, okay. I've been having some trouble with the equipment.
[de-escalating] [Navigator] Sometimes the reason is simple, like a busy schedule, low batteries, trouble opening the app, or not feeling well enough to take the readings.
[neutral] [User] Yes, I think it's been a mix of that.
[warm] [Navigator] If the equipment has been difficult to use, that is completely okay, and we can work through it together.
[neutral] [User] Alright, thank you.
[instruction] [Navigator] Please place the tablet and blood pressure monitor in front of you.
[neutral] [User] Okay, I have them here.
[instruction] [Navigator] Open the app and look for the button that says Take Blood Pressure under Your Tasks.
[neutral] [User] I see it now.
[instruction] [Navigator] Sit with your back supported, your feet flat on the floor, and your arm relaxed.
[neutral] [User] Okay, I'm ready.
[instruction] [Navigator] Start the reading and save it once it appears.
[neutral] [User] Alright.
[neutral] [User] The reading is done, and I saved it.
[instruction] [Navigator] Then place the pulse oximeter on one finger and save that reading when the numbers become steady.
[neutral] [User] Okay, I'm doing that now.
[neutral] [User] Alright, that one is saved too.
[calm] [Navigator] If you missed readings because of dizziness, weakness, shortness of breath, or any other new symptom, please let your care team know.
[neutral] [User] I will. I was feeling a little weak yesterday.
[de-escalating] [Navigator] If you feel frustrated with the equipment or have had repeated trouble, I can note that so someone can follow up and help.
[neutral] [User] Yes, that would be helpful.
[warm] [Navigator] These daily readings help your care team notice changes early, so getting back into a routine can make a real difference.
[neutral] [User] Okay, I understand.
[close] [Navigator] Once today's readings are complete, you should be back on track.
[neutral] [User] Alright, thank you.""",
    },
    {
        "title": "Triage - Neutral",
        "text": """[neutral] [Navigator] Thank you for calling the nurse triage line.
[neutral] [User] Hi.
[neutral] [Navigator] For quality and training purposes, this call may be recorded.
[neutral] [User] Okay.
[calm] [Navigator] If this is a medical emergency, please hang up and call 911 now.
[neutral] [User] No, it's not an emergency.
[verification] [Navigator] Please confirm your full name and your date of birth, including the year.
[neutral] [User] My name is David Miller, and my date of birth is August fourteenth, nineteen sixty-eight.
[neutral] [Navigator] I have August 14, 1968 on file.
[acknowledging] [Navigator] Thank you.
[neutral] [User] Yes.
[neutral] [Navigator] I understand you are calling about abdominal pain today.
[neutral] [User] That's right.
[neutral] [Navigator] Please tell me when the pain started and whether it began suddenly or built up gradually over time.
[neutral] [User] It started this morning and built up gradually.
[neutral] [Navigator] Tell me exactly where the pain is located.
[neutral] [User] It's mostly on the lower right side.
[neutral] [Navigator] On a scale from zero to ten, how severe is it right now?
[neutral] [User] About a six.
[neutral] [Navigator] Does the pain feel sharp, cramping, burning, pressure-like, or aching?
[neutral] [User] It feels more like cramping.
[neutral] [Navigator] Have you also had nausea, vomiting, fever, diarrhea, bloating, or trouble keeping fluids down?
[neutral] [User] I've had some nausea and bloating, but no vomiting.
[neutral] [Navigator] Have you noticed black stool, blood in stool, or vomit that looks dark like coffee grounds?
[neutral] [User] No.
[neutral] [Navigator] Is the pain so strong that it is hard for you to stand up straight or do normal activities?
[neutral] [User] No, but it is definitely uncomfortable.
[neutral] [Navigator] Have you had this kind of pain before?
[neutral] [User] No, not like this.
[neutral] [Navigator] Do you have any history of gallbladder problems, ulcers, bowel disease, or recent stomach infection?
[neutral] [User] No, I don't.
[neutral] [Navigator] Have you started any new medicine recently, or have you tried anything so far, such as rest, fluids, or over-the-counter medication?
[neutral] [User] I tried resting and drinking water, but that's all.
[close] [Navigator] Thank you. That gives the nurse a clearer picture of what is going on.
[neutral] [User] Okay.""",
    },
    {
        "title": "Triage - Warm and Reassuring",
        "text": """[neutral] [Navigator] Thank you for calling the triage line.
[neutral] [User] Hi.
[empathetic] [Navigator] I'm sorry you're not feeling well today.
[neutral] [User] Thank you.
[warm] [Navigator] I'll keep this simple and go step by step so the nurse has a clear picture of what is happening.
[neutral] [User] Okay.
[calm] [Navigator] If this is a medical emergency, please hang up and call 911 right away.
[neutral] [User] No, it's not an emergency.
[verification] [Navigator] First, please confirm your full name and date of birth, including the year.
[neutral] [User] My name is Linda Anderson, and my date of birth is May ninth, nineteen eighty-one.
[neutral] [Navigator] I have May 9, 1981 on file.
[acknowledging] [Navigator] Thank you.
[neutral] [User] Yes.
[warm] [Navigator] In your own words, please tell me what symptoms are bothering you most right now.
[neutral] [User] I've had a sore throat, body aches, fever, and I feel very tired.
[warm] [Navigator] Take your time.
[neutral] [User] Okay.
[empathetic] [Navigator] I know it can be hard to explain when you are not feeling well.
[neutral] [User] Yes, it's been a rough day.
[neutral] [Navigator] Are you having fever, chills, sore throat, cough, body aches, headache, fatigue, nausea, or dizziness?
[neutral] [User] Yes, I have fever, chills, sore throat, body aches, fatigue, and a little dizziness.
[neutral] [Navigator] When did these symptoms start, and which symptom came first?
[neutral] [User] They started yesterday, and the sore throat came first.
[neutral] [Navigator] Are they getting worse, staying the same, or starting to improve?
[neutral] [User] I think they're getting a little worse.
[neutral] [Navigator] Are you able to drink fluids and keep them down?
[neutral] [User] Yes, I can.
[neutral] [Navigator] Have you been able to stand and walk normally today?
[neutral] [User] Yes, but I feel weak.
[neutral] [Navigator] Have you checked your temperature, and if so, what was the highest reading?
[neutral] [User] Yes, it was one hundred one point four.
[neutral] [Navigator] Are you having any chest discomfort, shortness of breath, or wheezing?
[neutral] [User] No.
[neutral] [Navigator] Have you been around anyone who was sick recently, or have you had a recent flu or COVID exposure?
[neutral] [User] My grandson was sick earlier this week.
[neutral] [Navigator] Do you have asthma, diabetes, heart disease, or any other ongoing medical condition?
[neutral] [User] No, not that I know of.
[neutral] [Navigator] What medicines have you taken today to try to feel better?
[neutral] [User] Just Tylenol and some water.
[close] [Navigator] Thank you. These details will help the nurse decide the safest next step.
[neutral] [User] Okay, thank you.""",
    },
    {
        "title": "Triage - Urgent but Calm",
        "text": """[neutral] [Navigator] Thank you for calling the triage line.
[neutral] [User] Hello.
[calm] [Navigator] I need to ask a few quick safety questions.
[neutral] [User] Okay.
[calm] [Navigator] If you are having severe trouble breathing, chest pain that is getting worse, fainting, blue lips, sudden confusion, or symptoms of stroke, call 911 immediately.
[neutral] [User] Alright.
[verification] [Navigator] Please confirm your full name and date of birth.
[neutral] [User] My name is Robert Hayes, and my date of birth is January twenty-seventh, nineteen fifty-six.
[neutral] [Navigator] I have January 27, 1956 on file.
[acknowledging] [Navigator] Thank you.
[neutral] [User] Yes.
[instruction] [Navigator] If we get disconnected during this urgent review, I have your callback number as 303-555-4419. If there is a better number, please let me know.
[neutral] [User] No, that number is fine.
[urgent but calm] [Navigator] Please stay near your phone.
[neutral] [User] I will.
[calm] [Navigator] I understand this began after a new medication, and I want to check for signs of a serious reaction.
[neutral] [User] Yes, that's right.
[neutral] [Navigator] Are you having hives, a spreading rash, swelling of the lips or tongue, wheezing, trouble swallowing, chest tightness, dizziness, or feeling like you may pass out?
[neutral] [User] I have hives and some lip swelling, but I'm not wheezing.
[neutral] [Navigator] Please tell me the name of the medication, the dose, and the time you last took it.
[neutral] [User] It's amoxicillin, five hundred milligrams, and I took it about an hour ago.
[neutral] [Navigator] How long after taking it did the symptoms begin?
[neutral] [User] Maybe about twenty minutes later.
[neutral] [Navigator] Have you taken this medication before, or is this the first dose?
[neutral] [User] This was the first dose.
[neutral] [Navigator] Have you taken anything else today, such as another prescription, over-the-counter medicine, or supplement?
[neutral] [User] No, nothing else.
[calm] [Navigator] Because swelling or breathing symptoms after a new medication can become serious quickly, I am documenting this as urgent.
[neutral] [User] Okay.
[instruction] [Navigator] Please do not take another dose unless a clinician tells you to.
[neutral] [User] Alright.
[calm] [Navigator] Please stay near your phone and remain available for immediate nurse follow-up or transfer.
[neutral] [User] I will.""",
    },
]

PROMPTS: list[str] = [script["text"] for script in EXAMPLE_SCRIPTS]
PROMPTS_BY_SUFFIX = {f"{index:02d}": prompt.upper() for index, prompt in enumerate(PROMPTS)}
