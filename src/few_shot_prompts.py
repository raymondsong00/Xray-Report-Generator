import pandas as pd
import json
import sys

# reference:
# http://www.southsudanmedicaljournal.com/archive/2008-05/how-to-read-a-chest-x-ray-a-step-by-step-approach.html
step_by_step_inspection_checklist = '''Step 1 Airway:
Check trachea position, carina angle (60–100 degrees), and main stem bronchi.
Look for tubes, pacemaker wires, lines, and foreign bodies. Ensure correct positioning of endotracheal tubes.

Step 2 Bones and Soft Tissues:
Assess clavicles, ribs, thoracic spine, and humerus for fractures or lesions.
Examine soft tissues for subcutaneous air, foreign bodies, and surgical clips.
Be cautious of nipple shadows; compare both sides.

Step 3 Cardiac:
Evaluate heart size, borders, and for pneumomediastinum.
Inspect aorta for widening or calcification, heart valves, and major veins.

Step 4 Diaphragm:
Compare right to left hemidiaphragm height. Consider effusion or paralysis if discrepancies are noted.
Check for free air under the diaphragm in upright positions.

Step 5 Effusion:
Look for blunting of costophrenic angles and fissure tracking.
Examine pleura for thickening, loculations, calcifications, and pneumothorax.

Step 6 Lung Fields:
Identify infiltrates, noting their location and pattern (interstitial vs. alveolar).
Look for air bronchograms, tram tracking, nodules, Kerley B lines, and assess apices.

Step 7 Gastric Air Bubble:
Verify position, check for hiatus hernia, free air, and misplaced bowel loops.

Step 8 Hilum:
Evaluate position, size, lymph nodes, calcified nodules, mass lesions, and pulmonary artery size.
'''

def few_shot_train(old_json_fp:str, new_json_fp:str):
    data = None
    with open(old_json_fp, 'r') as old_json:
        data = json.load(old_json)
    new_qa = []
    for item in data:
        new_clue = 'following the X-ray reading procedures outlined as follows whenever you are not certain:\n\n' + step_by_step_inspection_checklist
        new_qa_item = item
        old_prompt_segments = item['conversations'][0]['value'].split(', ')
        old_question = old_prompt_segments[-1].strip()[:-1]
        old_question += ', which should closely resemble the style of the given AUTHOR?'
        old_prompt_segments[-1] = new_clue
        new_qa_item['conversations'][0]['value'] = ', '.join(old_prompt_segments) + '\n' + old_question
        new_qa.append(new_qa_item)
    with open(new_json_fp, 'w') as new_json:
        json.dump(new_qa, new_json, indent=4)

def few_shot_test(old_jsonl_fp:str, new_jsonl_fp:str):
    with open(new_jsonl_fp, 'w') as new_jsonl:
        with open(old_jsonl_fp, 'r') as old_jsonl:
            for line in old_jsonl:
                new_clue = 'following the X-ray reading procedures outlined as follows whenever you are not certain:\n\n' + step_by_step_inspection_checklist
                old_object = json.loads(line)
                new_object = old_object
                old_prompt_segments = old_object['text'].split(', ')
                old_question = old_prompt_segments[-1].strip()[:-1]
                old_question += ', which should closely resemble the style of the given AUTHOR?'
                old_prompt_segments[-1] = new_clue
                new_object['text'] = ', '.join(old_prompt_segments) + '\n' + old_question
                modified_line = json.dumps(new_object)
                new_jsonl.write(modified_line + '\n')

def main(selection:str):
    if selection == "generic":
        few_shot_train('./data/generic_prompt_train.json', './data/generic_prompt_few_shot_train.json')
        few_shot_test('./data/generic_prompt_test.jsonl', './data/generic_prompt_few_shot_test.jsonl')
    elif selection == "context":
        few_shot_train('./data/context_prompt_train.json', './data/context_prompt_few_shot_train.json')
        few_shot_test('./data/context_prompt_test.jsonl', './data/context_prompt_few_shot_test.jsonl')

if __name__=="__main__":
    selection = sys.argv[1]
    if selection not in ['generic', 'context']:
        raise("Please select either 'generic' or 'context'")
    main(selection)