SUPERCLRVER_sub_shape = {
    "car": ["suv", "wagon", "minivan", "sedan", "truck", "addi", "car"],
    "bus": ["articulated", "regular", "double", "school", "bus"],
    "motorbike": ["chopper", "dirtbike", "scooter", "cruiser", "motorbike"],
    "aeroplane": ["jet", "fighter", "biplane", "airliner", "aeroplane"],
    "bicycle": ["road", "utility", "mountain", "tandem", "bicycle"],
}

inverse_shape = {}
for key, value in SUPERCLRVER_sub_shape.items():
    for v in value:
        inverse_shape[v] = key
#todo 
# [done] modify default "is_correct" for spatial457: first do loose match, if true, return, else then do default is_correct
# [done] keep 2 instructions as hint (half open question) for spatial 457

# [done] implement is_correct for Omni-3D (metric for float gt -> MRA from omni3d paper page 5)
# design instruction hint for Omni-3D

#tsv index, image(base64)， question, answer, category(str)
#spatial 457 -> acc 0.12 -> pure open question
#spatial 457 -> acc ~0.4 -> open question with hint
#wxie -> acc 0.7(L1-L4) -> multichoice
import re
class Spatial457_utils:
    def __init__(self):

        return

    def get_random_answer(self, gt):
        import random

        all_attributes = {
            "size": ["small", "large"],
            "shape": [
                "airliner",
                "dirtbike",
                "road bike",
                "tandem bike",
                "suv",
                "wagon",
                "scooter",
                "mountain bike",
                "minivan",
                "sedan",
                "school bus",
                "fighter",
                "chopper",
                "double bus",
                "truck",
                "articulated bus",
                "cruiser",
                "jet",
                "utility bike",
                "regular bus",
                "biplane",
            ],
            "color": [
                "gray",
                "blue",
                "purple",
                "brown",
                "green",
                "cyan",
                "red",
                "yellow",
            ],
            "direction": ["left", "right", "front", "back"],
        }

        gt = gt.lower()
        if gt in ["yes", "no"]:
            return random.choice(["yes", "no"])
        if gt in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            return str(random.randint(0, 9))
        for key, value in all_attributes.items():
            if gt in value:
                return random.choice(value)

    def all_answers(self):
        all_attributes = {
            "size": ["small", "large"],
            "shape": [
                "airliner",
                "dirtbike",
                "road bike",
                "tandem bike",
                "suv",
                "wagon",
                "scooter",
                "mountain bike",
                "minivan",
                "sedan",
                "school bus",
                "fighter",
                "chopper",
                "double bus",
                "truck",
                "articulated bus",
                "cruiser",
                "jet",
                "utility bike",
                "regular bus",
                "biplane",
            ],
            "color": [
                "gray",
                "blue",
                "purple",
                "brown",
                "green",
                "cyan",
                "red",
                "yellow",
            ],
            "direction": ["left", "right", "front", "back"],
        }

        all_answers = ""
        for key, value in all_attributes.items():
            captical_value = [x.capitalize() for x in value]
            all_answers += ", ".join(captical_value) + ", "
        return all_answers.strip(", ")

    @staticmethod
    def loose_match(a, b):
        # Convert both inputs to string, trim spaces, and lowercase
        a = str(a).strip().lower()
        b = str(b).strip().lower()

        # Remove articles ('the', 'a', 'an')
        def remove_articles(s):
            return re.sub(r'\b(the|a|an)\b', '', s).strip()

        a = remove_articles(a)
        b = remove_articles(b)
        # Remove extra whitespace
        a = re.sub(r'\s+', ' ', a)
        b = re.sub(r'\s+', ' ', b)

        # Map common synonyms to standard values
        synonym_map = {
            "yes": "true",
            "no": "false",
            "correct": "true",
            "incorrect": "false",
            "right": "true",
            "wrong": "false"
        }
        a = synonym_map.get(a, a)
        b = synonym_map.get(b, b)

        return a == b

    def is_correct(self, answer, predict):
        if self.loose_match(answer, predict):
            return True
        
        text2num = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        predict = str(predict)
        answer = str(answer)

        if predict.lower() == "none":
            predict = "no"

        if predict.lower() == answer.lower():
            return True
        if predict == "0" and answer == "No":
            return True
        if predict.lower() in text2num and text2num[predict.lower()] == answer:
            return True
        if answer.lower() == "yes" and predict in [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
        ]:
            return True

        if self.category_correct(predict, answer):
            return True
        return False

    def category_correct(self, answer, gt_answer):
        answer = str(answer).lower().split(" ")[0]
        gt_answer = str(gt_answer).lower().split(" ")[0]

        if (
            answer in inverse_shape
            and gt_answer in inverse_shape
            and inverse_shape[answer] == inverse_shape[gt_answer]
        ):
            return True

        return False

    # is_correct for omni3d
    def is_correct_omni(self, level, answer, prediction):
        if "int" in level:
            try:
                aint = int(answer)
                pint = int(prediction)
                return aint == pint
            except ValueError:
                return False
        if "str" in level:
            return self.is_correct(answer, prediction)
        if "float" in level:
            try:
                y_hat = float(prediction)
                y = float(answer)
            except ValueError:
                return 0.
            #MRA
            C = [i / 100 * 50 + 0.5 for i in range(1,10)]
            sum_indicator = 0
            for theta in C:
                relative_error = abs(y_hat - y) / y
                indicator = 1 if relative_error < (1 - theta) else 0
                sum_indicator += indicator
            return sum_indicator / len(C)