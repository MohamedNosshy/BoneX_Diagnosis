from __future__ import division, print_function
import os
import numpy as np
# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
# Flask utils
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from heatmap import save_and_display_gradcam,make_gradcam_heatmap
from grad_cam_plus import save_and_display_gradcam_plusplus,compute_gradcam_plusplus



os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__, static_url_path='')

app.config['HEATMAP_FOLDER'] = 'heatmap','heatmapPlus'
app.config['UPLOAD_FOLDER'] = 'uploads'



# Load your trained models
MODEL_PATH_1 = 'D:/bahr/BoneX Diagnosis/Models/Spine/Spine.h5'
MODEL_PATH_2 = 'D:/bahr/BoneX Diagnosis/Models/Pelvis/pelvis.h5'
MODEL_PATH_3 = 'D:/bahr/BoneX Diagnosis/Models/Chest/Chest.h5'
MODEL_PATH_4 = 'D:/bahr/BoneX Diagnosis/Models/Elbow/elbow_mobilenet.h5'
MODEL_PATH_5 = 'D:/bahr/BoneX Diagnosis/Models/Finger/finger_mobilenet.h5'
MODEL_PATH_6 = 'D:/bahr/BoneX Diagnosis/Models/Forearm/Forearm_mobilenet.h5'
MODEL_PATH_7 = 'D:/bahr/BoneX Diagnosis/Models/Hand/hand_mobilenet.h5'
MODEL_PATH_8 = 'D:/bahr/BoneX Diagnosis/Models/Shoulder/shoulder_mobilenet.h5'
MODEL_PATH_9 = 'D:/bahr/BoneX Diagnosis/Models/Wrist/wrist_mobilenet.h5'
MODEL_PATH_NOTX = 'D:/bahr/BoneX Diagnosis/Models/Notx/NotX.h5'



model_notx = load_model(MODEL_PATH_NOTX)


model1 = load_model(MODEL_PATH_1)
model2 = load_model(MODEL_PATH_2)
model3 = load_model(MODEL_PATH_3)
model4 = load_model(MODEL_PATH_4)
model5 = load_model(MODEL_PATH_5)
model6 = load_model(MODEL_PATH_6)
model7 = load_model(MODEL_PATH_7)
model8 = load_model(MODEL_PATH_8)
model9 = load_model(MODEL_PATH_9)

def predict_xray(img_path):
    img = Image.open(img_path).resize((224, 224))

    # Preprocessing the image
    img = img.convert('RGB')
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32') / 255

    # Predict using the NotX model
    pred_notx = model_notx.predict(img)[0]

    # Assuming a binary classification (X-ray or Not X-ray)
    is_xray = pred_notx[0] < 0.5  # Adjust the threshold as needed

    return is_xray

class_dict = {
    'model1': {0: "dislocation",
               1: "fracture",
               2: "normal"},
    'model2': {0: "fracture",
               1: "normal",},
    'model3': {0:"flail",
               1:"hemothorax",
               2:"normal",
               3:"pneumothorax"},
    'model4': {0: "normal",
               1: "fracture",},
    'model5': {0: "normal",
               1: "fracture",},
    'model6': {0: "normal",
               1: "fracture",},
    'model7': {0: "normal",
               1: "fracture",},
    'model8': {0: "normal",
               1: "fracture",},
    'model9': {0: "normal",
               1: "fracture",},

}

disease_name = {
    'model1': {
        'dislocation': 'Spine Dislocation',
        'fracture': 'Spine Fracture',
        'normal': 'Normal'
    },
    'model2': {
        'fracture': 'Pelvis Fracture',
        'normal': 'Normal'
    },
    'model3': {
        'flail': 'Chest Flail',
        'hemothorax': 'Chest Hemothorax',
        'normal': 'Normal',
        'pneumothorax': 'Chest Pneumothorax'
    },
    'model4': {
        'fracture': 'Elbow Fracture',
        'normal': 'Normal'
    },
    'model5': {
        'fracture': 'Finger Fracture',
        'normal': 'Normal'
    },
    'model6': {
        'fracture': 'Forearm Fracture',
        'normal': 'Normal'
    },
    'model7': {
        'fracture': 'Hand Fracture',
        'normal': 'Normal'
    },
    'model8': {
        'fracture': 'Shoulder Fracture',
        'normal': 'Normal'
    },
    'model9': {
        'fracture': 'Wrist Fracture',
        'normal': 'Normal'
    },

}


#definition of disease

definition_dict = {
    'model1': {
        'dislocation': 'A spinal dislocation, often referred to as a dislocated spine, is a severe and potentially life-threatening injury that occurs when one or more vertebrae in the spine are displaced from their normal positions, causing disruption to the spinal cord and surrounding structures. It is important to note that spinal dislocations are relatively rare and are typically the result of high-energy trauma',
        'fracture': 'A spinal fracture is a break or disruption in one or more of the bones (vertebrae) in the spine. These fractures can vary widely in severity, location, and potential complications',
    },
    'model2': {
        'fracture': 'A pelvic fracture refers to a break or fracture in one or more of the bones that make up the pelvic girdle. The pelvis is a sturdy, ring-like structure consisting of several bones, including the ilium, ischium, pubis, and sacrum. Pelvic fractures can vary in severity, from stable fractures that may not cause significant displacement to unstable fractures that can lead to severe bleeding and damage to nearby organs.',
    },
    'model3': {
        'flail': 'Flail chest is a serious medical condition that occurs when a segment of the rib cage becomes detached from the rest of the chest wall due to multiple rib fractures. This condition results in a paradoxical chest movement during breathing, which can lead to significant respiratory distress and potential life-threatening complications. ',
        'hemothorax': 'A hemothorax is a medical condition that occurs when blood accumulates in the pleural cavity, the space between the lungs and the chest wall. This condition can lead to impaired lung function and potentially life-threatening complications.',
        'pneumothorax': 'A pneumothorax is a medical condition characterized by the presence of air in the pleural cavity, the space between the lung and the chest wall. This air accumulation can compress the lung and interfere with its ability to expand properly, leading to respiratory distress and potentially life-threatening complications.'
    },
    'model4':{
        'fracture':'An elbow fracture refers to a break or crack in one or more of the bones that make up the elbow joint, including the humerus, radius, and ulna.',
        },
    'model5':{
        'fracture':'A finger fracture involves a break or crack in one or more of the bones in the fingers',
        },
    'model6':{
        'fracture':'A forearm fracture involves a break or crack in one or both of the bones of the forearm, the radius, and/or ulna.',        
        },
    'model7':{
        'fracture':'A hand fracture involves a break or crack in one or more of the bones in the hand, including the metacarpals and phalanges.',   
        },
    'model8':{
        'fracture':'A shoulder fracture involves a break or crack in one or more of the bones that make up the shoulder, including the clavicle (collarbone), scapula (shoulder blade), or humerus (upper arm bone).',       
        },
    'model9':{
        'fracture':'A wrist fracture involves a break or crack in one or more of the bones in the wrist, commonly the radius or ulna.',
        }
}


#causes of disease

cause1 = {
    'model1': {
        'dislocation':'Traumatic Dislocation: This occurs as a result of a sudden injury, such as a car accident, sports-related injury, or a fall. Traumatic dislocations can be partial or complete, depending on the extent of vertebral misalignment.' , 
        'fracture': 'Trauma: Spine fractures are commonly caused by accidents, falls, sports injuries, and other traumatic events that exert excessive force on the spine.',
    },
    'model2': {
        'fracture': 'Trauma: The most common cause of pelvic fractures is high-impact trauma, such as motor vehicle accidents, falls from heights, sports injuries, or motorcycle accidents.',
    },
    'model3': {
        'flail': 'Trauma: Chest flail is usually the result of a traumatic event, such as a car accident, motorcycle crash, industrial accident, or significant blunt force injury to the chest.',
        'hemothorax': 'Trauma: The most common cause of chest hemothorax is trauma or injury, such as a car accident, fall, or penetrating wound (e.g., gunshot or stab wound), which can damage blood vessels, leading to bleeding into the pleural cavity.',
        'pneumothorax': 'Trauma: The most common cause of pneumothorax is trauma or injury to the chest, such as a rib fracture, car accident, fall, or penetrating wound (e.g., gunshot or stab wound).',
    },
    'model4':{
        'fracture':'Trauma: Direct impact or force to the elbow, often from a fall or accident.',
        },
    'model5':{
        'fracture':'Trauma: Direct impact or force to the finger, often from crushing, slamming, or a fall.',
        },
    'model6':{
        'fracture':'Trauma: Direct impact or force to the forearm, often from falls or accidents.',
        },
    'model7':{
        'fracture':'Trauma: Direct impact or force to the hand, often from crushing, slamming, or a fall.',
        },
    'model8':{
        'fracture':'Trauma: Direct impact or force to the shoulder, often from falls, accidents, or sports injuries.',
        },
    'model9':{
        'fracture':'Falls: Landing on an outstretched hand can result in a wrist fracture.',
        }
}

cause2 = {
    'model1': {
        'dislocation':'Congenital Dislocation: Some individuals are born with spine abnormalities that can lead to dislocations. These are typically present from birth and may require surgical intervention.' , 
        'fracture': 'Osteoporosis: Weakened bones due to osteoporosis can make vertebrae more susceptible to compression fractures, even from minor trauma or simple movements.',
    },
    'model2': {
        'fracture': 'Osteoporosis: In some cases, especially in elderly individuals with weakened bones, minor falls or trauma can result in pelvic fractures.',
    },
    'model3': {
        'flail': 'Rib Fractures: Multiple rib fractures, typically involving two or more adjacent ribs in at least two places, are necessary to create a flail segment.',
        'hemothorax': 'Medical Procedures: In some cases, chest hemothorax can occur as a complication of medical procedures, such as lung biopsies or thoracentesis (removal of pleural fluid).',
        'pneumothorax': 'Spontaneous Pneumothorax: In some cases, a pneumothorax can occur spontaneously, without any apparent injury or underlying lung disease. This is more common in tall, thin individuals and is often linked to the rupture of small air-filled sacs in the lung called "blebs" or "bullae."',
    },
    'model4':{
        'fracture':'Overuse or Repetitive Stress: Continuous stress on the elbow joint, common in certain sports or activities.',
        },
    'model5':{
        'fracture':'Sports Injuries: Common in activities where fingers are at risk, such as basketball or rock climbing.',
        },
    'model6':{
        'fracture':'Sports Injuries: Common in activities where the forearm is at risk, such as contact sports or biking.',
        },
    'model7':{
        'fracture':'Sports Injuries: Common in activities where the hand is at risk, such as basketball, football, or martial arts.',
        },
    'model9':{
        'fracture':'Sports Injuries: Particularly in activities where there is a risk of falling on the hand.',
        }
}

cause3 = {
    'model1': {
        'dislocation':'Degenerative Dislocation: Over time, the spine can undergo changes due to wear and tear, resulting in dislocation. Conditions like spondylolisthesis and facet joint arthropathy can contribute to this type of dislocation.' , 
        'fracture': 'Tumors: Cancerous or benign tumors that develop in or around the spine can weaken the vertebrae and increase the risk of fractures.',
    },
    'model2': {
        'fracture': 'Pathological Fractures: Fractures can occur in individuals with pre-existing bone diseases or tumors in the pelvis.',
    },
    'model3': {
        'flail': 'High-Energy Impact: The force required to cause a flail chest is often substantial, and the injury can be associated with other internal injuries.',
        'hemothorax': 'Medical Conditions: Certain medical conditions, such as a bleeding disorder, pulmonary embolism, or cancer, can also lead to spontaneous hemothorax.',
        'pneumothorax': 'Medical Procedures: Pneumothorax can occur as a complication of medical procedures, such as lung biopsies or mechanical ventilation.',
    },
    'model9':{
        'fracture':'Trauma: Direct impact or force to the wrist.',
        }
}

cause4 = {
    'model3': {
        'pneumothorax': 'Lung Disease: Certain lung diseases, such as chronic obstructive pulmonary disease (COPD) or cystic fibrosis, can increase the risk of developing a pneumothorax.',
    },

}

#Symptoms of disease

symptom1 = {
    'model1': {
        'dislocation': 'Severe back pain',
        'fracture': 'Pain at the site of the fracture, which may be sudden and severe.',
    },
    'model2': {
        'fracture': 'Severe pelvic or lower abdominal pain',
    },
    'model3': {
        'flail': 'Severe chest pain.',
        'hemothorax': 'Sudden, sharp chest pain on the side of the bleeding.',
        'pneumothorax': 'Sudden, sharp chest pain, which is often felt on one side.'
    },
    'model4':{
        'fracture':'Pain',
        },
    'model5':{
        'fracture':'Pain',
        },
    'model6':{
        'fracture':'Pain',
        },
    'model7':{
        'fracture':'Pain',
        },
    'model8':{
        'fracture':'Pain',
        },
    'model9':{
        'fracture':'Pain',
        }
}

symptom2 = {
    'model1': {
        'dislocation': 'Limited range of motion',
        'fracture': 'Limited range of motion.',
    },
    'model2': {
        'fracture': 'Difficulty or inability to walk or bear weight on the affected leg(s).',
    },
    'model3': {
        'flail': 'Paradoxical Chest Wall Movement: The flail segment moves in the opposite direction of the rest of the chest wall during breathing.',
        'hemothorax': 'Difficulty breathing or shortness of breath.',
        'pneumothorax': 'Difficulty breathing or shortness of breath'
    },
    'model4':{
        'fracture':'Swelling',
        },
    'model5':{
        'fracture':'Swelling',
        },
    'model6':{
        'fracture':'Swelling',
        },
    'model7':{
        'fracture':'Swelling',
        },
    'model8':{
        'fracture':'Swelling',
        },
    'model9':{
        'fracture':'Swelling',
        }
}


symptom3 = {
    'model1': {
        'dislocation': 'Numbness or tingling in the arms or legs',
        'fracture': 'Neurological symptoms such as numbness, tingling, or weakness if the fracture affects the spinal cord or nerves.',
    },
    'model2': {
        'fracture': 'Bruising, swelling, or tenderness over the pelvic area.',
    },
    'model3': {
        'flail': 'Difficulty Breathing: Shallow, rapid, and painful breathing.',
        'hemothorax': 'Rapid heart rate (tachycardia).',
        'pneumothorax': 'Rapid heart rate (tachycardia).'
    },
    'model4':{
        'fracture':'Bruising',
        },
    'model5':{
        'fracture':'Bruising',
        },
    'model6':{
        'fracture':'Bruising',
        },
    'model7':{
        'fracture':'Bruising',
        },
    'model8':{
        'fracture':'Bruising',
        },
    'model9':{
        'fracture':'Bruising',
        }
}


symptom4 = {
    'model1': {
        'dislocation': 'Muscle weakness',
        'fracture': 'Changes in posture or height (compression fractures can lead to a stooped appearance).',
    },
    'model2': {
        'fracture': 'Pain or numbness in the groin, hips, or thighs.',
    },
    'model3': {
        'flail': 'Bruising and Swelling: The affected area may appear bruised and swollen.',
        'hemothorax': 'Decreased breath sounds on one side of the chest during auscultation.',
        'pneumothorax': 'Decreased breath sounds on the affected side during auscultation.'
    },
    'model4':{
        'fracture':'Limited range of motion',
        },
    'model5':{
        'fracture':'Difficulty moving the finger',
        },
    'model6':{
        'fracture':'Difficulty moving or using the forearm',
        },
    'model7':{
        'fracture':'Difficulty moving the hand or fingers',
        },
    'model8':{
        'fracture':'Limited range of motion',
        },
    'model9':{
        'fracture':'Difficulty moving the wrist or hand',
        }
}


symptom5 = {
    'model1': {
        'dislocation': 'Difficulty walking or maintaining balance',
    },
    'model2': {
        'fracture': 'Abdominal or pelvic distention.',
    },
    'model3': {
        'flail': 'Reduced Lung Function: Impaired oxygen exchange due to the inability of the affected portion of the chest to expand properly.',
        'hemothorax': 'Bluish skin or lips (cyanosis) in severe cases, indicating a lack of oxygen.',
        'pneumothorax': 'Bluish skin or lips (cyanosis) in severe cases, indicating a lack of oxygen.'
    },
    'model4':{
        'fracture':'Deformity',
        },
    'model5':{
        'fracture':'Deformity',
        },
    'model6':{
        'fracture':'Deformity',
        },
    'model7':{
        'fracture':'Deformity',
        },
    'model8':{
        'fracture':'Deformity',
        },
    'model9':{
        'fracture':'Deformity',
        }
}

symptom6 = {
    'model1': {
        'dislocation': 'Bowel or bladder dysfunction (in severe cases)',
    },
    'model2': {
        'fracture': 'Hematuria (blood in the urine), which may indicate bladder or urethral injury in severe fractures.',
    },
}


#treatment of disease

treatment1 = {
    'model1': {
        'dislocation': 'Conservative Management: Mild cases may be managed with rest, pain management, physical therapy, and the use of braces or immobilization devices.',
        'fracture': 'Conservative Management: Compression fractures without significant spinal cord involvement can often be managed conservatively with rest, pain management, bracing, and physical therapy. In cases related to osteoporosis, medications to strengthen bones may be prescribed.',
    },
    'model2': {
        'fracture': 'Pain Management: Medications to manage pain and discomfort.',
    },
    'model3': {
        'flail': 'Pain Management: Pain relief is essential. Medications, including opioids, may be administered to manage pain.',
        'hemothorax': 'Chest Tube Insertion: A chest tube is often inserted into the pleural cavity to drain the accumulated blood and restore normal lung function. The tube may remain in place until the bleeding has stopped, and the chest cavity is clear of blood.',
        'pneumothorax': 'Observation: Small, asymptomatic pneumothoraxes may be observed closely without immediate intervention, especially if they are not expanding.'
    },
    'model4':{
        'fracture':'Immobilization: Splinting or casting to prevent movement and promote healing.',
        },
    'model5':{
        'fracture':'Immobilization: Splinting or casting to restrict movement and support healing.',
        },
    'model6':{
        'fracture':'Immobilization: Casting or splinting to prevent movement and aid in healing.',
        },
    'model7':{
        'fracture':'Immobilization: Splinting or casting to restrict movement and support healing.',
        },
    'model8':{
        'fracture':'Immobilization: Sling or brace to restrict movement and aid in healing.',
        },
    'model9':{
        'fracture':'Immobilization: Splinting or casting to restrict movement and support healing.',
        }
}


treatment2 = {
    'model1': {
        'dislocation': 'Surgical Intervention: Severe dislocations, especially those causing neurological deficits or instability, often require surgery. Procedures may involve realigning the vertebrae, fusing them together, or using implants to stabilize the spine.',
        'fracture': 'Surgical Intervention: Surgical options may be considered for unstable fractures, fractures with neurological deficits, or fractures that fail to heal with conservative treatment. Surgery may involve stabilizing the spine with instrumentation like rods and screws or performing a vertebral augmentation procedure like kyphoplasty or vertebroplasty.',
    },
    'model2': {
        'fracture': 'Stable Fractures: Stable fractures may be managed conservatively with rest, limited weight-bearing, and assistive devices like crutches or a walker.',
    },
    'model3': {
        'flail': 'Oxygen Therapy: Oxygen may be provided to maintain oxygen levels in the bloodstream.',
        'hemothorax': 'Blood Transfusion: In cases of significant blood loss, blood transfusions may be necessary to replace lost blood and improve oxygen delivery to the body tissues.',
        'pneumothorax': 'Chest Tube Insertion: If the pneumothorax is large, symptomatic, or not improving on its own, a chest tube is typically inserted into the pleural cavity to remove the trapped air and allow the lung to re-expand. The tube may be connected to a suction device to facilitate air removal.'
    },
    'model4':{
        'fracture':'Medication: Pain relievers and anti-inflammatory drugs as prescribed.',
        },
    'model5':{
        'fracture':'Medication: Pain relievers and anti-inflammatory drugs as prescribed.',
        },
    'model6':{
        'fracture':'Pain Management: Medications prescribed for pain relief and inflammation.',
        },
    'model7':{
        'fracture':'Medication: Pain relievers and anti-inflammatory drugs as prescribed.',
        },
    'model8':{
        'fracture':'Medication: Pain relievers and anti-inflammatory drugs as prescribed.',
        },
    'model9':{
        'fracture':'Medication: Pain relievers and anti-inflammatory drugs as prescribed.',
        }
}


treatment3 = {
    'model1': {
        'dislocation': 'Rehabilitation: After surgery or conservative treatment, rehabilitation is essential to regain strength, flexibility, and function.',
        'fracture': ' You might need to wear a back brace to hold your spine in alignment and help your broken vertebrae heal properly',
    },
    'model2': {
        'fracture': 'Unstable Fractures: Unstable fractures often require surgical intervention to realign and stabilize the bones. Surgical options may include the use of plates, screws, rods, or external fixators.',
    },
    'model3': {
        'flail': 'Chest Tube Placement: If there is evidence of a collapsed lung (pneumothorax) or blood in the chest cavity (hemothorax), a chest tube may be inserted to relieve pressure and drain fluids.',
        'hemothorax': 'Treatment of Underlying Cause: If the hemothorax is due to an underlying medical condition, such as a bleeding disorder or cancer, treatment will be directed at addressing that condition.',
        'pneumothorax': 'Oxygen Therapy: Oxygen therapy may be administered to help increase the oxygen levels in the bloodstream, particularly in cases of partial pneumothorax or mild symptoms.'
    },
    'model4':{
        'fracture':'Physical Therapy: Exercises to improve strength and range of motion.',
        },
    'model5':{
        'fracture':'Elevation: Keeping the hand elevated to reduce swelling.',
        },
    'model6':{
        'fracture':'Elevation: Keeping the forearm elevated to reduce swelling.',
        },
    'model7':{
        'fracture':'Elevation: Keeping the hand elevated to reduce swelling.',
        },
    'model8':{
        'fracture':'Physical Therapy: Exercises to regain strength and mobility.',
        },
    'model9':{
        'fracture':'Elevation: Keeping the hand elevated to reduce swelling.',
        }
}


treatment4 = {
    'model2': {
        'fracture': 'Blood Transfusion: In cases of severe bleeding, blood transfusions may be necessary to replace lost blood volume.',
    },
    'model3': {
        'flail': 'Intubation and Mechanical Ventilation: In severe cases, mechanical ventilation may be required to support breathing.',
        'hemothorax': 'Surgery: In some cases, particularly when other treatments are unsuccessful or if there is severe damage to the chest wall or blood vessels, surgery may be required to control bleeding and repair any injuries.',
        'pneumothorax': 'Treatment of Underlying Cause: If the pneumothorax is due to an underlying lung disease or condition, such as COPD, treatment will be directed at managing that condition.'
    },
    'model4':{
        'fracture':'Surgery: In severe cases or complex fractures.',
        },
    'model5':{
        'fracture':'Buddy Taping: Taping the injured finger to an adjacent one for support.',
        },
    'model6':{
        'fracture':'Physical Therapy: Rehabilitation exercises to restore strength and mobility.',
        },
    'model7':{
        'fracture':'Buddy Taping: Taping the injured finger to an adjacent one for support.',
        },
    'model8':{
        'fracture':'Surgery: In some cases, especially for complex fractures or if bones are displaced.',
        },
    'model9':{
        'fracture':'Physical Therapy: Exercises to regain strength and flexibility.',
        }
}


treatment5 = {
    'model2': {
        'fracture': 'Treatment of Associated Injuries: Addressing any associated injuries to nearby structures, such as the bladder, urethra, or blood vessels, is essential.',
    },
    'model3': {
        'flail': 'Surgical Stabilization: In some cases, especially when other treatments are not effective or if the flail chest is causing severe respiratory distress, surgical stabilization of the fractured ribs with plates or screws may be considered.',
    },
    'model6':{
        'fracture':'Surgery: In more severe cases or if bones are displaced and need realignment.',
        },
    'model7':{
        'fracture':'Physical Therapy: Exercises to regain strength and flexibility.',
        },
    'model9':{
        'fracture':'Surgery: In some cases, especially for complex fractures or if bones are displaced.',
        }
}


treatment6 = {
    'model3': {
        'flail': 'Monitoring and Support: Close monitoring in an intensive care unit (ICU) is often necessary to manage potential complications and provide critical care.',
    },
}


#Complications of disease

complication1 = {
    'model1': {
        'dislocation': 'Spinal Cord Injury: Severe dislocations can lead to spinal cord compression or injury, potentially causing permanent neurological deficits.',
        'fracture': 'Neurological deficits, including paralysis, if the spinal cord is damaged.',
    },
    'model2': {
        'fracture': 'Internal bleeding.',
    },
    'model3': {
        'flail': 'Pneumonia: Due to impaired lung function.',
        'hemothorax': 'Complications of untreated or severe chest hemothorax can include infection, lung collapse (atelectasis), scarring (fibrosis) of the pleura, and potentially life-threatening conditions such as hypovolemic shock due to significant blood loss.',
        'pneumothorax': 'Untreated or severe pneumothorax can lead to complications such as infection, tension pneumothorax (a life-threatening condition where air continues to accumulate, causing increased pressure in the chest cavity and compression of the heart and other organs), and respiratory distress.'
    },
    'model4':{
        'fracture':'Stiffness: Reduced flexibility in the joint.',
        },
    'model5':{
        'fracture':'Stiffness: Reduced flexibility in the finger.',
        },
    'model6':{
        'fracture':'Compartment Syndrome: Swelling causes increased pressure within the forearm compartments, potentially affecting blood flow.',
        },
    'model7':{
        'fracture':'Stiffness: Reduced flexibility in the hand.',
        },
    'model8':{
        'fracture':'Stiffness: Reduced flexibility in the shoulder.',
        },
    'model9':{
        'fracture':'Stiffness: Reduced flexibility in the wrist.',
        }
}


complication2 = {
    'model1': {
        'dislocation': 'Chronic Pain: Even after treatment, some individuals may experience chronic back pain or discomfort.',
        'fracture': 'Chronic pain or disability.',
    },
    'model2': {
        'fracture': 'Nerve damage.',
    },
    'model3': {
        'flail': 'Respiratory Distress Syndrome: A serious complication that can occur if the chest flail is not adequately managed.',
        'hemothorax': 'leading to respiratory failure (inability to breathe properly) ',
        'pneumothorax': 'severe hypotension (obstructive shock) and even death.'
    },
    'model4':{
        'fracture':'Malunion: Improper healing leading to misalignment.',
        },
    'model5':{
        'fracture':'Malunion: Misalignment during healing.',
        },
    'model6':{
        'fracture':'Nerve or Blood Vessel Damage: Risk of injury to nerves or blood vessels near the fracture site.',
        },
    'model7':{
        'fracture':'Malunion: Misalignment during healing.',
        },
    'model8':{
        'fracture':'Nerve or Blood Vessel Damage: Risk of injury to surrounding structures.',
        },
    'model9':{
        'fracture':'Nerve or Blood Vessel Damage: Risk of injury to surrounding structures.',
        }
}


complication3 = {
    'model1': {
        'dislocation': 'Mobility Issues: Mobility and flexibility may be compromised, especially in cases of fusion surgery.',
        'fracture': 'Reduced mobility and quality of life.',
    },
    'model2': {
        'fracture': 'Bladder or urethral injury.',
    },
    'model3': {
        'flail': 'Long-Term Pain: Chronic chest pain may persist even after the fractures have healed.',
    },
    'model4':{
        'fracture':'Infection: Risk post-surgery or with open fractures.',
        },
    'model5':{
        'fracture':'Infection: Risk post-surgery or with open fractures.',
        },
    'model7':{
        'fracture':'Infection: Risk post-surgery or with open fractures.',
        },
    'model8':{
        'fracture':'Chronic Pain: Some individuals may experience ongoing pain.',
        },
}


complication4 = {
    'model2': {
        'fracture': 'Infections, particularly in open fractures.',
    },
    'model3': {
        'flail': 'Other Traumatic Injuries: Chest flail often occurs alongside other traumatic injuries, which may also require treatment.',
    },
}


complication5 = {
    'model2': {
        'fracture': 'Long-term pain and disability.',
    },
}


#advice to patient

advice_dict = {
    'model1': {
        'dislocation': 'Spine dislocations are serious medical conditions that require timely diagnosis and appropriate treatment. The prognosis and recovery can vary widely depending on the extent of the dislocation and the success of treatment. Consulting with a healthcare professional or spine specialist is essential for proper evaluation and management.',
        'fracture': 'The treatment approach for spine fractures is highly individualized and depends on the specific circumstances of the injury and the patient is overall health. Early diagnosis and appropriate treatment are crucial to achieving the best possible outcomes and minimizing long-term complications. A spine specialist, typically an orthopedic surgeon or neurosurgeon, plays a central role in the evaluation and management of spine fractures.',
    },
    'model2': {
        'fracture': 'Pelvic fractures can be serious injuries, particularly in the case of unstable fractures. Prompt evaluation and appropriate treatment are essential to minimize complications and improve outcomes. Patients with pelvic fractures are typically managed by a multidisciplinary medical team, which may include trauma surgeons, orthopedic surgeons, and other specialists as needed.',
    },
    'model3': {
        'flail': 'Chest flail is a critical medical condition that requires immediate medical attention and intervention. Prompt diagnosis and appropriate treatment are crucial to improve the patient is chances of recovery and reduce the risk of life-threatening complications. Chest flail is often managed by a team of healthcare professionals, including trauma surgeons, respiratory therapists, and critical care specialists.',
        'hemothorax': 'Chest hemothorax is a medical emergency that requires prompt evaluation and treatment. If you or someone you know experiences symptoms such as chest pain, difficulty breathing, or rapid heart rate after trauma or injury, seek immediate medical attention. Early intervention can significantly improve outcomes and prevent complications.',
        'pneumothorax': 'Chest pneumothorax is a medical emergency that requires prompt evaluation and treatment, especially if it causes significant symptoms or impairs lung function. Seek immediate medical attention if you suspect you or someone else has a pneumothorax, particularly after trauma or injury. Early intervention can be critical for a positive outcome.'
    },
    'model4':{
        'fracture':'Follow your doctors instructions for rest and rehabilitation .  Attend follow-up appointments to monitor progress. Report any unusual pain, swelling, or changes in symptoms. Engage in prescribed physical therapy exercises.  Use assistive devices as recommended',
        },
    'model5':{
        'fracture':'Rest and Protect , Follow Medical Advice , Ice , Elevate , Report Changes',
        },
    'model6':{
        'fracture':'Follow Medical Instructions , Rest and Protect , Report Changes  , Physical Therapy',
        },
    'model7':{
        'fracture':'Follow Medical Instructions , Rest and Protect , Report Changes , Ice and Elevation',
        },
    'model8':{
        'fracture':'Follow Medical Instructions , Rest and Protect , Report Changes , Ice and Elevation',
        },
    'model9':{
        'fracture':'Follow Medical Instructions , Rest and Protect , Report Changes , Ice and Elevation',
        }
}


def get_class_label(model_type, key):
    return class_dict.get(model_type, {}).get(key, "Unknown")

@app.route('/uploads/<filename>')
def upload_img(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle POST request if needed
        return render_template('index.html')
    else:
        # Handle GET request
        return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def splash():
    if request.method == 'POST':
        model_type = request.form['model_type']
        return render_template('splash.html', model_type=model_type)
    return render_template('splash.html', model_type='')



def model_predict(img_path, model, model_type):
    img = Image.open(img_path).resize((224, 224))

    # Preprocessing the image
    img = img.convert('RGB')
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32') / 255

    preds = model.predict(img)[0]

    prediction = sorted(
        [(get_class_label(model_type, i), round(j * 100, 2)) for i, j in enumerate(preds)],
        reverse=True,
        key=lambda x: x[1]
    )

    return prediction, img


def model_predict_not_xray(img_path, model):
    img = Image.open(img_path).resize((224, 224))

    # Preprocessing the image
    img = img.convert('RGB')
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32') / 255

    preds = model.predict(img)[0]

    prediction = sorted(
        [(get_class_label('model10', i), round(j * 100, 2)) for i, j in enumerate(preds)],
        reverse=True,
        key=lambda x: x[1]
    )

    return prediction


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        model_type = request.form['model_type']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        file_name = os.path.basename(file_path)
        
        is_xray = predict_xray(file_path)


        if model_type == 'model1':
            model_to_use = model1
            last_conv_layer_name = "Conv_1"
            class_labels = class_dict['model1']
        elif model_type == 'model2':
            model_to_use = model2
            last_conv_layer_name = "Conv_1"
            class_labels = class_dict['model2']
        elif model_type == 'model3':
            model_to_use = model3
            last_conv_layer_name = "Conv_1"
            class_labels = class_dict['model3']
        elif model_type == 'model4':
            model_to_use = model4
            last_conv_layer_name = "Conv_1"
            class_labels = class_dict['model4']
        elif model_type == 'model5':
            model_to_use = model5
            last_conv_layer_name = "Conv_1"
            class_labels = class_dict['model5']    
        elif model_type == 'model6':
            model_to_use = model6
            last_conv_layer_name = "Conv_1"
            class_labels = class_dict['model6']
        elif model_type == 'model7':
            model_to_use = model7
            last_conv_layer_name = "Conv_1"
            class_labels = class_dict['model7']
        elif model_type == 'model8':
            model_to_use = model8
            last_conv_layer_name = "Conv_1"
            class_labels = class_dict['model8']
        elif model_type == 'model9':
            model_to_use = model9
            last_conv_layer_name = "Conv_1"
            class_labels = class_dict['model9']  

        else:
            return "Invalid model type"

        pred, img = model_predict(file_path, model_to_use, model_type)
        predicted_class_label = pred[0][0]  # Get the top predicted class label

        
        x_ray = "X-ray" if is_xray else "Not X-ray"



        
        disease_text = disease_name.get(model_type, {}).get(predicted_class_label, 'not available')

        # Retrieve the definition text based on the predicted class label
        definition_text = definition_dict.get(model_type, {}).get(predicted_class_label, 'not available')
        
        # Retrieve the causes text based on the predicted class label
        cause1_text = cause1.get(model_type, {}).get(predicted_class_label, 'not available')
        cause2_text = cause2.get(model_type, {}).get(predicted_class_label, 'not available')
        cause3_text = cause3.get(model_type, {}).get(predicted_class_label, 'not available')
        cause4_text = cause4.get(model_type, {}).get(predicted_class_label, 'not available')
        
        # Retrieve the symptoms text based on the predicted class label
        symptom1_text = symptom1.get(model_type, {}).get(predicted_class_label, 'not available')
        symptom2_text = symptom2.get(model_type, {}).get(predicted_class_label, 'not available')
        symptom3_text = symptom3.get(model_type, {}).get(predicted_class_label, 'not available')
        symptom4_text = symptom4.get(model_type, {}).get(predicted_class_label, 'not available')
        symptom5_text = symptom5.get(model_type, {}).get(predicted_class_label, 'not available')
        symptom6_text = symptom6.get(model_type, {}).get(predicted_class_label, 'not available')
        
        # Retrieve the treatments text based on the predicted class label
        treatment1_text = treatment1.get(model_type, {}).get(predicted_class_label, 'not available')
        treatment2_text = treatment2.get(model_type, {}).get(predicted_class_label, 'not available')
        treatment3_text = treatment3.get(model_type, {}).get(predicted_class_label, 'not available')
        treatment4_text = treatment4.get(model_type, {}).get(predicted_class_label, 'not available')
        treatment5_text = treatment5.get(model_type, {}).get(predicted_class_label, 'not available')
        treatment6_text = treatment6.get(model_type, {}).get(predicted_class_label, 'not available')
        
        # Retrieve the Complications text based on the predicted class label
        complication1_text = complication1.get(model_type, {}).get(predicted_class_label, 'not available')
        complication2_text = complication2.get(model_type, {}).get(predicted_class_label, 'not available')
        complication3_text = complication3.get(model_type, {}).get(predicted_class_label, 'not available')
        complication4_text = complication4.get(model_type, {}).get(predicted_class_label, 'not available')
        complication5_text = complication5.get(model_type, {}).get(predicted_class_label, 'not available')
        
        # Retrieve the advice text based on the predicted class label
        advice_text = advice_dict.get(model_type, {}).get(predicted_class_label, 'not available')
        
        heatmapPlus = compute_gradcam_plusplus(model_to_use, img, last_conv_layer_name)
        fnamePlus = save_and_display_gradcam_plusplus(file_path, heatmapPlus)
        
        
        heatmap = make_gradcam_heatmap(img, model_to_use, last_conv_layer_name)
        fname = save_and_display_gradcam(file_path, heatmap)

        return render_template('predict.html', file_name=file_name, heatmap_file=fname, heatmap_filePlus=fnamePlus, result=pred[0][1],
                               definition=definition_text,
                               cause1=cause1_text, cause2=cause2_text, cause3=cause3_text, cause4=cause4_text,
                               symptom1 = symptom1_text , symptom2 = symptom2_text , symptom3 = symptom3_text , symptom4 = symptom4_text , symptom5 = symptom5_text , symptom6 = symptom6_text ,
                               treatment1 = treatment1_text , treatment2 = treatment2_text , treatment3 = treatment3_text , treatment4 = treatment4_text , treatment5 = treatment5_text , treatment6 = treatment6_text,
                               complication1 = complication1_text , complication2 = complication2_text , complication3 = complication3_text , complication4 = complication4_text , complication5 = complication5_text ,
                               advice=advice_text,
                               disease=disease_text,
                               x_ray = x_ray,
                               )


    return render_template('predict.html', file_name='', heatmap_file='', result='',
                           definition='',
                           cause1='', cause2='', cause3='', cause4='',
                           symptom1='', symptom2='', symptom3='', symptom4='', symptom5='', symptom6='',
                           treatment1='', treatment2='', treatment3='', treatment4='', treatment5='', treatment6='',
                           complication1='', complication2='', complication3='', complication4='', complication5='',
                           advice='',
                           disease='',
                           x_ray='',
                           )
@app.route('/information', methods=['GET', 'POST'])
def information():
    if request.method == 'POST':
        # Handle POST request if needed
        return render_template('information.html')
    else:
        # Handle GET request
        return render_template('information.html')
    
@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        # Handle POST request if needed
        return render_template('form.html')
    else:
        # Handle GET request
        return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True, host="localhost", port=8080)















