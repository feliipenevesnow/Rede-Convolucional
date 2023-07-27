import cv2
import mediapipe as mp
import numpy as np


line_thickness = 25  # Espessura da linha desenhada
line_color = (255, 255, 255)  # Cor da linha (branco)


def gera_imagem():
  prev_x, prev_y = None, None  # Coordenadas anteriores do indicador


  cap = cv2.VideoCapture(0)
  mpHands = mp.solutions.hands
  hands = mpHands.Hands()
  mpDraw = mp.solutions.drawing_utils


  window_name = "Desenhando"  # Nome da janela de desenho
  cv2.namedWindow(window_name)


  # Criar uma imagem preta para desenhar a linha
  frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  drawing_img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)  # Fundo preto


  # Define os pontos das pontas dos dedos que serão utilizados para o reconhecimento de gestos
  finger_tip_ids = [4, 8, 12, 16, 20]


  # Mapeamento dos números desenhados com base nos dedos levantados
  number_mapping = {
      (1, 0, 0, 0, 0): 0,
      (1, 1, 0, 0, 0): 1,
      (1, 1, 1, 0, 0): 2,
      (1, 1, 1, 1, 0): 3,
      (1, 1, 1, 1, 1): 4,
      (0, 1, 1, 1, 1): 5,
      (0, 0, 1, 1, 1): 6,
      (0, 0, 0, 1, 1): 7,
      (0, 0, 0, 0, 1): 8,
      (0, 0, 0, 0, 0): 9,
  }


  while True:
      success, frame = cap.read()
      frame = cv2.flip(frame, 1)  # Espelha o quadro horizontalmente
      frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      results = hands.process(frameRGB)
      number = 1


      if results.multi_hand_landmarks:
          for handLms in results.multi_hand_landmarks:
              finger_states = []
              cx, cy = None, None


              for id, lm in enumerate(handLms.landmark):
                  h, w, c = frame.shape
                  cx, cy = int(lm.x * w), int(lm.y * h)


                  if id in finger_tip_ids:
                      # Verifica se o dedo está levantado com base na posição vertical (eixo y)
                      finger_states.append(1 if cy < handLms.landmark[id - 2].y * h else 0)


                  if id == 8:
                      cv2.circle(frame, (cx, cy), 25, (255, 0, 255), cv2.FILLED)


                  number = number_mapping.get(tuple(finger_states), None)
                  if number is not None:
                      print("Número desenhado:", number)
                      print(type(number))


              if number == 1 and prev_x is not None and prev_y is not None:
                  draw_line(drawing_img, prev_x, prev_y, cx, cy)


              elif number == 2:
                  drawing_img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)  # Limpa a imagem de desenho


              prev_x, prev_y = (cx, cy) if number == 1 else (None, None)


              mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)


          cv2.imshow("Output", frame)
          cv2.imshow(window_name, drawing_img)  # Exibe a imagem de desenho


      key = cv2.waitKey(1)


      if number is None:  # Pressione 's' para sair e salvar a imagem com a linha desenhada
          cv2.imwrite("desenho.png", drawing_img)
          print("Imagem salva como 'desenho.png'")
          break


  cap.release()
  cv2.destroyAllWindows()




def draw_line(img, x1, y1, x2, y2):
  cv2.line(img, (x1, y1), (x2, y2), line_color, line_thickness)




gera_imagem()

