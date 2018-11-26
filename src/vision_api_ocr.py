import os
import cfg
import cv2
import json
import math
import itertools
import sys

import numpy as np
import pandas as pd

from google.cloud import vision
from google.cloud.vision import types
from Levenshtein import *
from google.protobuf.json_format import MessageToJson


class VisionAPIOCRLocationBased:
    
    def __init__(self, image):
        # import google app credential
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cfg.google_app_credential
        
        # init client
        self.client = vision.ImageAnnotatorClient()
        self.image = image
        
        # init ktp keys
        self.KTP_KEY = {'nik': 'NIK', 
           'name': 'Nama', 
           'dateOfBirth': 'Tempat / Tgl Lahir',
           'gender': 'Jenis Kelamin',
           'address': 'Alamat', 
           'rtrw': 'RT RW',
           'kelurahan': 'Kel Desa',
           'kecamatan': 'Kecamatan',
           'religion': 'Agama', 
           'maritalStatus': 'Status Perkawinan',
           'occupation': 'Pekerjaan'}
        
    def get_responce_vision_api(self, image, scaler=1):
        """
        Param: cv2.image or numpy array
        Return: response from google vision api (dict)
        """
        
        height, width = image.shape[:2]
        
        # if big -> doing downsampling until it's small enough to be sent to vision api
        while True:
            image = cv2.resize(image, (int(width*scaler), int(height*scaler)))
            success, encoded_image = cv2.imencode('.png', image)
            content = encoded_image.tobytes()

            if sys.getsizeof(content) > 10485760:
                scaler = scaler*0.8
            else:
                break
                
        image_to_vision = types.Image(content=content)
        # getting response
        response = self.client.document_text_detection(image=image_to_vision)

        return response
    
    def get_ktp_angle(self, response):   
        """
        Param: vision api response (dict)
        Return: angle (float)
        """
        
        try:
            box_angle = []
            # find the mean angle of text boxes
            for item in response['textAnnotations']:
                vertices = item['boundingPoly']['vertices']

                length_x_1 = abs(vertices[1]['x'] - vertices[0]['x'])
                length_y_1 = abs(vertices[1]['y'] - vertices[0]['y'])
                grad_1 = length_y_1/length_x_1
                angle_1 = math.degrees(math.atan(grad_1))

                length_x_2 = abs(vertices[3]['x'] - vertices[2]['x'])
                length_y_2 = abs(vertices[3]['y'] - vertices[2]['y'])
                grad_2 = length_y_2/length_x_2
                angle_2 = math.degrees(math.atan(grad_2))

                box_angle.append([angle_1, angle_2])

            return np.mean(box_angle)
        
        except:
            print("Error on finding ktp angle -> ", vertices)
            return None
        
    def check_if_close(self, vertices_one, vertices_two, angle):
        """
        Param: box 1 and 2 (dict), angle (float)
        Return: whether two text boxes are in one line (bool)
        """
        
        if angle is None:
            angle = 0
        expected_add_y = int(math.tan(math.radians(angle))*(vertices_two[0]['x']-vertices_one[0]['x']))

        if vertices_one[1]['y']+expected_add_y < vertices_two[1]['y'] + (vertices_two[2]['y'] - vertices_two[1]['y'])/2 \
        and vertices_one[1]['y']+expected_add_y > vertices_two[1]['y'] - (vertices_two[2]['y'] - vertices_two[1]['y'])/2:
            return True
        else:
            return False
        
    def find_value_position(self, inline_vertices):
        """
        Param: array of inline boxes (list)
        Return: where the value is (list)
        """
        
        list_mean_dist = []
        vertices_pos_x = inline_vertices[:,:,0]
        vertices_pos_y = inline_vertices[:,:,1]

        min_y = np.mean([((i[2]-i[1])+(i[3]-i[0]))/2 for i in vertices_pos_y])
        #print(min_y)

        for i in range(len(vertices_pos_x)):
            if i != 0:
                dist = np.array(sorted(vertices_pos_x[i])[:2]) - np.array(sorted(vertices_pos_x[i-1])[2:])
                list_mean_dist.append(np.mean(dist))
        key_value_mark = [0]+[0 if score<min_y else 1 for score in list_mean_dist]
        split = np.argmax(key_value_mark)

        return split
    
    def extract_city_province(self, string):
        """
        Param: list of possible city/province text (list)
        Return: city or province (string)
        """
        
        sentence = string.copy()
        distance_kota = [distance(i, 'KOTA') for i in sentence]
        distance_kabupaten = [distance(i, 'KABUPATEN') for i in sentence]
        distance_provinsi = [distance(i, 'PROVINSI') for i in sentence]
        
        # finding whether it's provice, city or kabupaten
        if np.min(distance_kota) <= 1:
            string.pop(np.argmin(distance_kota))
            return 'city', ' '.join(string)
        elif np.min(distance_kabupaten) <= 3:
            string.pop(np.argmin(distance_kabupaten))
            return 'city', ' '.join(string)
        elif np.min(distance_provinsi) <= 2:
            string.pop(np.argmin(distance_provinsi))
            return 'province', ' '.join(string)
    
    def extract_gender_blood_type(self, string):
        """
        Param: list of possible gender and bloodtype text (list)
        Return: gender and bloodtype (string)
        """
        
        sentence = string.copy()
        darah_idx, gol_idx = 999, 999
        for word in sentence:
            if distance('Darah', word) <= 2:
                darah_idx = string.index(word)
            if distance('Gol', word) <= 2:
                gol_idx = string.index(word)
        try:
            string_gender = string[:min(darah_idx, gol_idx)]
        except:
            string_gender = ''
        try:
            string_blood_type = string[min(darah_idx, gol_idx):][-1]
            if string_blood_type == '+' or string_blood_type == '-':
                string_blood_type = ''.join(string[min(darah_idx, gol_idx):][-2:])
        except:
            string_blood_type = ''

        if string_blood_type == '0':
            string_blood_type = 'O'
        elif string_blood_type == '0+':
            string_blood_type = 'O+'
        elif string_blood_type == '0-':
            string_blood_type == 'O-'

        if string_blood_type in ['A', 'A+', 'A-', 'AB', 'AB+', 'AB-', 'B', 'B+', 'B-', 'O', 'O+', 'O-']:
            blood_type = string_blood_type
        else:
            blood_type = '-'

        if distance(' '.join(string_gender), 'LAKI LAKI') <= 3:
            gender ='LAKI-LAKI'
        elif distance(' '.join(string_gender), 'PEREMPUAN') <= 3:
            gender = 'PEREMPUAN'
        else:
            gender = '-'

        return {'gender': gender, 
                'blood_type': blood_type}
    
    def extract_religion(self, string):
        """
        Param: list of possible religion text (list)
        Return: religion (string)
        """
        
        sentence = string.copy()
        for word in sentence:
            if distance('Agama', word) <= 2:
                string.pop(string.index(word))

        if distance(' '.join(string), 'ISLAM') <= 3:
            return 'ISLAM'
        elif distance(' '.join(string), 'KRISTEN') <= 3:
            return 'KRISTEN'
        elif distance(' '.join(string), 'KHATOLIK') <= 3:
            return 'KHATOLIK'
        elif distance(' '.join(string), 'BUDHA') <= 3:
            return 'BUDHA'
        elif distance(' '.join(string), 'HINDU') <= 3:
            return 'HINDU'
        elif distance(' '.join(string), 'KONG HU CU') <= 3:
            return 'KONG HU CU'
        else:
            return '-'
        
    def extract_marital_status(self, string):
        """
        Param: list of possible marital status text (list)
        Return: marital status (string)
        """
        
        sentence = string.copy()
        for word in sentence:
            if distance('Status', word) <= 2:
                string.pop(string.index(word))
            if distance('Perkawinan', word) <= 3:
                string.pop(string.index(word))

        if distance(' '.join(string), 'KAWIN') <= 3:
            return 'KAWIN'
        elif distance(' '.join(string), 'BELUM KAWIN') <= 3:
            return 'BELUM KAWIN'
        elif distance(' '.join(string), 'CERAI HIDUP') <= 3:
            return 'CERAI HIDUP'
        elif distance(' '.join(string), 'CERAI MATI') <= 3:
            return 'CERAI MATI'
        else:
            return '-'
        
    def extract_occupation(self, string):
        """
        Param: list of possible occupation text (list)
        Return: occupation (string)
        """
        
        occupation_list = ["Belum/Tidak Bekerja", "Mengurus Rumah Tangga", "Pelajar/Mahasiswa", "Pensiunan", "Pegawai Negeri Sipil (PNS)",
                           "Tentara Nasional Indonesia (TNI)", "Kepolisian RI", "Perdagangan", "Petani/Pekebun", "Peternak", "Nelayan/Perikanan",
                           "Industri", "Konstruksi", "Transportasi", "Karyawan Swasta", "Karyawan BUMN", "Karyawan BUMD", "Karyawan Honorer",
                           "Buruh Harian Lepas", "Buruh Tani/Perkebunan", "Buruh Nelayan/Perikanan", "Buruh Peternakan", "Pembantu Rumah Tangga",
                           "Tukang Cukur", "Tukang Listrik", "Tukang Batu", "Tukang Kayu", "Tukang Sol Sepatu", "Tukang Las/Pandai Besi", "Tukang Jahit",
                           "Penata Rambut", "Penata Rias", "Penata Busana", "Mekanik", "Tukang Gigi", "Seniman", "Tabib", "Paraji", "Perancang Busana",
                           "Penterjemah", "Imam Masjid", "Pendeta", "Pastur", "Wartawan", "Ustadz/Mubaligh", "Juru Masak", "Promotor Acara", "Anggota DPR-RI",
                           "Anggota DPD", "Anggota BPK", "Presiden", "Wakil Presiden", "Anggota Mahkamah Konstitusi", "Anggota Kabinet/Kementerian",
                           "Duta Besar", "Gubernur", "Wakil Gubernur", "Bupati", "Wakil Bupati", "Walikota", "Wakil Walikota", "Anggota DPRD Propinsi",
                           "Anggota DPRD Kabupaten/Kota", "Dosen", "Guru", "Pilot", "Pengacara", "Notaris", "Arsitek", "Akuntan", "Konsultan", "Dokter",
                           "Bidan", "Perawat", "Apoteker", "Psikiater/Psikolog", "Penyiar Televisi", "Penyiar Radio", "Pelaut", "Peneliti", "Sopir",
                           "Pialang", "Paranormal", "Pedagang", "Perangkat Desa", "Kepala Desa", "Biarawati", "Wiraswasta"]
        
        sentence = string.copy()
        for word in sentence:
            if distance('Pekerjaan', word) <= 2:
                string.pop(string.index(word))
            if word.isdigit():
                string.pop(string.index(word))

        for occ in occupation_list:
            occ = occ.upper()
            if distance(occ, ' '.join(string)) <= 2:
                return occ

        return '-'
    
    def extract_text(self, items, angle):
        """
        Param: items of text annotated (list), angle (float)
        Return: kay value pair (dict)
        """
        
        dict_key = {}
        dict_pos_key = {}
        string_dict = {}
        key_dict = {}
        key_value_pair = {}
        not_catched_string = []
        
        # init list_ktp_key, later will be pop-ed out one by one
        list_ktp_key = list(self.KTP_KEY.keys())

        for i, this in enumerate(items):
            if this['description'] not in list(itertools.chain.from_iterable(dict_key.values())):
                list_similar = {}
                list_pos_similar = {}

                list_similar[this['boundingPoly']['vertices'][0]['x']] = this['description']
                list_pos_similar[this['boundingPoly']['vertices'][0]['x']] = [(i['x'], i['y']) for i in this['boundingPoly']['vertices']]
                
                # find which 'that' text boxes are inline with this
                for that in items:
                    if this['description'] != that['description'] and that['description'] not in list(itertools.chain.from_iterable(dict_key.values())):
                        if self.check_if_close(this['boundingPoly']['vertices'], that['boundingPoly']['vertices'], angle):
                            list_similar[that['boundingPoly']['vertices'][0]['x']] = that['description']
                            list_pos_similar[that['boundingPoly']['vertices'][0]['x']] = [(i['x'], i['y']) for i in that['boundingPoly']['vertices']]

                # inline text and its location on pixel
                join_text = list(dict(sorted(list_similar.items())).values())
                join_loc = list(dict(sorted(list_pos_similar.items())).values())
                
                # which index the value located (:thatindex => key)
                idx_value = self.find_value_position(np.array(join_loc))

                # assign text and location on a dict
                dict_key[this['boundingPoly']['vertices'][0]['y']] = join_text
                dict_pos_key[this['boundingPoly']['vertices'][0]['y']] = join_loc
                
                # make a string of key and value pair
                string_dict[this['boundingPoly']['vertices'][0]['y']] = "".join([
                    c if c not in '`~@#$%^&*()\}]|[{'";:?.><!·•" else '' for c in " ".join(join_text[idx_value:])]).strip()
                key_dict[this['boundingPoly']['vertices'][0]['y']] = "".join([
                    c if c not in '`~@#$%^&*()\}]|[{'";:?.><!·•" else '' for c in " ".join(join_text[:idx_value])]).strip()

                key = "".join([c if c not in '`~@#$%^&*()\}]|[{'";:?.><!·•" else '' for c in " ".join(join_text[:idx_value])]).strip()

                #print(idx_value, join_text, key, join_text[idx_value:])
                
                # if one line of text boxes have no key, check if it's city or province
                if idx_value == 0:
                    not_catched_string.append(join_text[idx_value:])
                    try:
                        which_key, value = self.extract_city_province(join_text[idx_value:])
                        key_value_pair[which_key] = value
                    except:
                        pass
                else:
                    if key != '' and len(list_ktp_key) >= 1:
                        calculate_dist = [distance(self.KTP_KEY[i], key) for i in list_ktp_key]
                        if np.min(calculate_dist) <= 4:
                            which_key = list_ktp_key[np.argmin(calculate_dist)]
                            # check if gender
                            if which_key == 'gender':
                                tmp_value = self.extract_gender_blood_type(join_text[idx_value:])
                                key_value_pair['bloodType'] = tmp_value['blood_type']
                                value = tmp_value['gender']
                            # check if religion
                            elif which_key == 'religion':
                                value = self.extract_religion(join_text[idx_value:])
                            # check if marital status
                            elif which_key == 'maritalStatus':
                                value = self.extract_marital_status(join_text[idx_value:])
                            # check if occupation
                            elif which_key == 'occupation':
                                value = self.extract_occupation(join_text[idx_value:])
                            else:
                                value = "".join([
                                    c if c not in '`~@#$%^&*()\}]|[{'";:?.><!·•" else '' for c in " ".join(join_text[idx_value:])]).strip()
                            key_value_pair[which_key] = value
                            
                            # pop everything out, so that we know which key that's not available
                            list_ktp_key.pop(list_ktp_key.index(which_key))

        # hardcoded this value
        key_value_pair['nationality'] = 'WNI'
        key_value_pair['validUntil'] = 'SEUMUR HIDUP'

        # check if there's key with empty value
        if len(list_ktp_key) >= 1:
            for text in not_catched_string:
                # try if it's marital status but in one line
                if 'maritalStatus' in list_ktp_key and \
                (np.min([distance(i, 'Status') for i in text]) <= 2 \
                 or np.min([distance(i, 'Perkawinan') for i in text]) <= 3):
                    value = self.extract_marital_status(text)
                    key_value_pair['maritalStatus'] = value
                    list_ktp_key.pop(list_ktp_key.index('maritalStatus'))

            for which_key in list_ktp_key:
                key_value_pair[which_key] = '-'

        #print(list_ktp_key)

        return key_value_pair
    
    def get_text(self):
        # get vision API reponse given KTP image
        response = self.get_responce_vision_api(self.image)
        
        # convert API response object to json, then to dict
        json_result = MessageToJson(response)
        dict_result = json.loads(json_result)
        
        # find angle
        angle = self.get_ktp_angle(dict_result)
        
        # extract text and return the key:value pair of dict
        items = dict_result['textAnnotations'][1:]
        text = self.extract_text(items, angle)
        
        return text