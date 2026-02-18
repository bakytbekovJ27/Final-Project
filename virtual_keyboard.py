"""
OpenCV Virtual Keyboard with Gesture Control 
Виртуальная клавиатура с управлением жестами
Требования: pip install opencv-python mediapipe numpy
"""
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import webbrowser
from urllib.parse import quote

# ============================================================================
# ИНИЦИАЛИЗАЦИЯ
# ============================================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

ret, frame = cap.read()
if not ret:
    print("❌ Ошибка: не удалось получить доступ к камере")
    exit()

frame_height, frame_width, _ = frame.shape
print(f"INFO: Камера запущена с разрешением {frame_width}x{frame_height}")

# ============================================================================
# НАСТРОЙКИ КЛАВИАТУРЫ
# ============================================================================

# Раскладки клавиатуры
KEYBOARD_LAYOUTS = {
    'en_main': [
        ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
        ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
        ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
        ['z', 'x', 'c', 'v', 'b', 'n', 'm', 'Delete']
    ],
    'sym_main': [
        ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
        ['#', '%', ':', ',', '.', ';', '(', ')', '_'],
        ['@', '#', '$', '&', '*', '-', '+', '=', '/'],
        ['[', ']', '{', '}', '<', '>', '?', '!', 'Delete']
    ]
}

# ============================================================================
# ПОИСКОВЫЕ СИСТЕМЫ С АВТООТПРАВКОЙ
# ============================================================================
SEARCH_ENGINES = {
    'Google': {
        'url': 'https://www.google.com/search?q=',
        'color': (66, 133, 244)
    },
    'YouTube': {
        'url': 'https://www.youtube.com/results?search_query=',
        'color': (0, 0, 255)
    },
    'Yandex': {
        'url': 'https://yandex.ru/search/?text=',
        'color': (255, 0, 0)
    },
    'Perplexity': {
        'url': 'https://www.perplexity.ai/search?q=',
        'color': (32, 201, 172)
    }
}

# Состояние приложения
current_layout = 'en_main'
current_language = 'EN' 
current_search_engine = 'Google'
text_input = ""
last_click_time = 0
click_delay = 0.5
show_dropdown = False
dropdown_positions = {}
pinch_was_active = False

# ============================================================================
# ФУНКЦИИ УТИЛИТЫ
# ============================================================================

def calculate_distance(point1, point2, width, height):
    """Вычисляет расстояние между двумя точками"""
    x1, y1 = int(point1.x * width), int(point1.y * height)
    x2, y2 = int(point2.x * width), int(point2.y * height)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def is_pinch_gesture(hand_landmarks, frame_width, frame_height):
    """Определяет жест 'щипок' (указательный + большой палец соприкасаются)"""
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    
    distance = calculate_distance(index_tip, thumb_tip, frame_width, frame_height)
    pinch_x = int(index_tip.x * frame_width)
    pinch_y = int(index_tip.y * frame_height)
    
    return distance < 50, (pinch_x, pinch_y)

def draw_rounded_rectangle(img, pt1, pt2, color, thickness=-1, radius=12):
    """Рисует скругленный прямоугольник в стиле Apple"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    if thickness == -1:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

def draw_dropdown_menu(frame, anchor_rect):
    """Рисует выпадающее меню выбора поисковой системы (вверх от кнопки)"""
    global dropdown_positions
    dropdown_positions = {}
    
    anchor_x, anchor_y, anchor_w, _ = anchor_rect
    
    dropdown_width = 220
    item_height = 50
    num_items = len(SEARCH_ENGINES)
    dropdown_height = num_items * item_height
    
    dropdown_x = anchor_x
    dropdown_y = anchor_y - dropdown_height - 10
    
    # Тень
    shadow_offset = 3
    shadow = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
    cv2.rectangle(shadow, 
                 (dropdown_x + shadow_offset, dropdown_y + shadow_offset),
                 (dropdown_x + dropdown_width + shadow_offset, 
                  dropdown_y + dropdown_height + shadow_offset),
                 (0, 0, 0, 100), -1)
    
    # Фон меню
    draw_rounded_rectangle(frame,
                          (dropdown_x, dropdown_y),
                          (dropdown_x + dropdown_width, dropdown_y + dropdown_height),
                          (255, 255, 255), -1, 10)
    
    # Граница
    draw_rounded_rectangle(frame,
                          (dropdown_x, dropdown_y),
                          (dropdown_x + dropdown_width, dropdown_y + dropdown_height),
                          (204, 204, 204), 2, 10)
    
    # Элементы меню
    for idx, (engine_name, engine_data) in enumerate(SEARCH_ENGINES.items()):
        item_y = dropdown_y + idx * item_height
        
        # Подсветка выбранного
        if engine_name == current_search_engine:
            draw_rounded_rectangle(frame,
                                  (dropdown_x + 5, item_y + 5),
                                  (dropdown_x + dropdown_width - 5, item_y + item_height - 5),
                                  (0, 122, 255), -1, 8)
            text_color = (255, 255, 255)
        else:
            text_color = (0, 0, 0)
        
        dropdown_positions[engine_name] = (dropdown_x, item_y, dropdown_width, item_height)
        
        # Цветной индикатор
        cv2.circle(frame, (dropdown_x + 20, item_y + item_height // 2), 6, 
                  engine_data['color'], -1)
        
        # Название
        cv2.putText(frame, engine_name, 
                   (dropdown_x + 40, item_y + item_height // 2 + 7),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Разделитель
        if idx < len(SEARCH_ENGINES) - 1:
            cv2.line(frame, 
                    (dropdown_x + 10, item_y + item_height),
                    (dropdown_x + dropdown_width - 10, item_y + item_height),
                    (230, 230, 230), 1)

def draw_keyboard(frame, layout_key, y_start):
    """Рисует клавиатуру в минималистичном стиле Apple"""
    layout = KEYBOARD_LAYOUTS[layout_key]
    
    key_width = 80
    key_height = 70
    key_margin = 10
    row_margin = 12
    
    key_positions = {}
    current_y = y_start
    
    for row_idx, row in enumerate(layout):
        num_keys = len(row)
        
        total_margin_width = (num_keys - 1) * key_margin
        available_width = frame_width * 0.95
        current_key_width = int((available_width - total_margin_width) / num_keys)
        current_key_width = min(current_key_width, 100)

        total_width = num_keys * current_key_width + (num_keys - 1) * key_margin
        start_x = (frame_width - total_width) // 2
        
        for key_idx, key in enumerate(row):
            x = start_x + key_idx * (current_key_width + key_margin)
            y = current_y
            
            if key == 'Delete':
                temp_width = int(current_key_width * 1.2)
                x = start_x + key_idx * (current_key_width + key_margin) - (temp_width - current_key_width)
                draw_width = temp_width
            else:
                draw_width = current_key_width
            
            key_rect = (x, y, draw_width, key_height)
            key_positions[key] = key_rect
            
            if key == 'Delete':
                key_color = (120, 60, 60)
            else:
                key_color = (255, 255, 255)
            
            draw_rounded_rectangle(frame, 
                                  (x, y), 
                                  (x + draw_width, y + key_height),
                                  key_color, -1, 8)
            
            draw_rounded_rectangle(frame, 
                                  (x, y), 
                                  (x + draw_width, y + key_height),
                                  (224, 224, 224), 2, 8)
            
            display_text = key
            text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            text_x = x + (draw_width - text_size[0]) // 2
            text_y = y + (key_height + text_size[1]) // 2
            
            text_color = (255, 255, 255) if key == 'Delete' else (0, 0, 0)
            cv2.putText(frame, display_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
        
        current_y += key_height + row_margin
    
    # Нижний ряд с крупными кнопками
    bottom_row_y = current_y + 5
    bottom_key_height = key_height + 10
    
    lang_display = current_language
    space_display = 'space'
    search_selector_display = current_search_engine.split(' ')[0]
    search_selector_color = SEARCH_ENGINES[current_search_engine]['color']
    
    bottom_keys = [
        ('SEARCH_SEL', 140, search_selector_color, search_selector_display),
        ('LANG', 100, (255, 255, 255), lang_display),
        ('SPACE', 350, (255, 255, 255), space_display),
        ('.', 80, (255, 255, 255), '.'),
        ('SEND', 120, (0, 122, 255), 'Send')
    ]
    
    total_bottom_width = sum([w for _, w, _, _ in bottom_keys]) + (len(bottom_keys) - 1) * key_margin
    bottom_x = (frame_width - total_bottom_width) // 2
    
    for key_name, width, color, display in bottom_keys:
        key_rect = (bottom_x, bottom_row_y, width, bottom_key_height)
        key_positions[key_name] = key_rect
        
        draw_rounded_rectangle(frame, 
                              (bottom_x, bottom_row_y),
                              (bottom_x + width, bottom_row_y + bottom_key_height),
                              color, -1, 10)
        
        border_color = (224, 224, 224) if key_name not in ['SEND', 'SEARCH_SEL'] else color
        draw_rounded_rectangle(frame, 
                              (bottom_x, bottom_row_y),
                              (bottom_x + width, bottom_row_y + bottom_key_height),
                              border_color, 2, 10)
        
        if key_name == 'SEARCH_SEL':
            r, g, b = color
            if (r*0.299 + g*0.587 + b*0.114) > 186:
                text_color = (0, 0, 0)
            else:
                text_color = (255, 255, 255)
        elif key_name == 'SEND':
            text_color = (255, 255, 255)
        else:
            text_color = (0, 0, 0)
            
        text_size = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = bottom_x + (width - text_size[0]) // 2
        text_y = bottom_row_y + (bottom_key_height + text_size[1]) // 2
        
        cv2.putText(frame, display, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        bottom_x += width + key_margin
    
    return key_positions

def draw_search_bar(frame):
    """Рисует строку поиска в минималистичном стиле"""
    bar_height = 80
    bar_y = 40
    
    shadow_offset = 2
    draw_rounded_rectangle(frame,
                          (20 + shadow_offset, bar_y + shadow_offset),
                          (frame_width - 20 + shadow_offset, bar_y + bar_height + shadow_offset),
                          (200, 200, 200), -1, 12)
    
    draw_rounded_rectangle(frame,
                          (20, bar_y),
                          (frame_width - 20, bar_y + bar_height),
                          (255, 255, 255), -1, 12)
    
    draw_rounded_rectangle(frame,
                          (20, bar_y),
                          (frame_width - 20, bar_y + bar_height),
                          (204, 204, 204), 2, 12)
    
    input_x = 40
    
    if text_input:
        display_text = text_input
        text_color = (0, 0, 0)
    else:
        display_text = "Enter search text..."
        text_color = (150, 150, 150)
    
    max_chars = 80
    if len(display_text) > max_chars:
        display_text = "..." + display_text[-(max_chars-3):]
    
    cv2.putText(frame, display_text, (input_x + 15, bar_y + bar_height // 2 + 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
    
    return {
        'input': (input_x, bar_y + 10, frame_width - 80, bar_height - 20)
    }

def check_key_press(x, y, key_positions):
    """Проверяет, какая клавиша нажата"""
    for key, (kx, ky, kw, kh) in key_positions.items():
        if kx <= x <= kx + kw and ky <= y <= ky + kh:
            return key
    return None

def check_dropdown_click(x, y):
    """Проверяет клик по dropdown меню"""
    for engine_name, (dx, dy, dw, dh) in dropdown_positions.items():
        if dx <= x <= dx + dw and dy <= y <= dy + dh:
            return engine_name
    return None

def perform_search():
    """Выполняет поиск в выбранной системе с автоотправкой"""
    global text_input
    if not text_input.strip():
        print("⚠️ Текст пуст, нечего искать")
        return
    
    engine = SEARCH_ENGINES[current_search_engine]
    
    print(f"🔎 Поиск в {current_search_engine}: {text_input}")
    
    try:
        url = engine['url'] + quote(text_input)
        print(f"🌐 URL: {url}")
        webbrowser.open(url)
        print("✅ Браузер открыт!")
            
    except Exception as e:
        print(f"❌ Ошибка открытия браузера: {e}")
    
    text_input = ""

def handle_key_press(key):
    """Обрабатывает нажатие клавиши"""
    global current_layout, current_language, text_input, last_click_time
    
    current_time = time.time()
    
    if current_time - last_click_time < click_delay:
        return
    
    last_click_time = current_time
    
    if key == 'SPACE':
        text_input += ' '
        print("⌨️  Пробел")
    elif key == 'Delete':
        text_input = text_input[:-1]
        print("⌨️  Удаление")
    elif key == 'LANG':
        if current_language == 'EN':
            current_language = 'SYM'
            current_layout = 'sym_main'
            print("⌨️  Режим: Символы")
        else:
            current_language = 'EN'
            current_layout = 'en_main'
            print("⌨️  Language: English")
    elif key == 'SEND':
        perform_search()
    elif key == 'SEARCH_SEL':
        pass
    else:
        text_input += key
        print(f"⌨️  Нажата: '{key}' | Текст: {text_input}")

# ============================================================================
# ГЛАВНЫЙ ЦИКЛ
# ============================================================================

print("=" * 70)
print("   ВИРТУАЛЬНАЯ КЛАВИАТУРА С УПРАВЛЕНИЕМ ЖЕСТАМИ")
print("   OpenCV Virtual Keyboard with Gesture Control")
print("=" * 70)
print("УПРАВЛЕНИЕ:")
print("  🤏 Соедините указательный и большой палец для нажатия")
print("  👆 Щипок на кнопке поисковика - открыть меню выбора")
print("\nДОСТУПНЫЕ ПОИСКОВИКИ:")
for engine_name in SEARCH_ENGINES.keys():
    print(f"  ✓ {engine_name}")
print("\nКЛАВИШИ:")
print("  Q или ESC - Выход")
print("  C - Очистить текст")
print("=" * 70)

keyboard_y_start = frame_height - 500

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    search_bar_areas = draw_search_bar(frame)
    key_positions = draw_keyboard(frame, current_layout, keyboard_y_start)
    if show_dropdown:
        anchor_rect = key_positions.get('SEARCH_SEL')
        if anchor_rect:
            draw_dropdown_menu(frame, anchor_rect)
    
    pinch_active = False
    pinch_pos = None
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 122, 255), thickness=2)
            )
            
            is_pinching, pinch_position = is_pinch_gesture(hand_landmarks, frame_width, frame_height)
            
            if is_pinching:
                pinch_active = True
                pinch_pos = pinch_position
                x, y = pinch_pos
                
                cv2.circle(frame, (x, y), 25, (0, 255, 255), 3)
                cv2.circle(frame, (x, y), 12, (0, 255, 255), -1)
                
                if not pinch_was_active:
                    current_time = time.time()
                    
                    if current_time - last_click_time >= click_delay:
                        if show_dropdown:
                            selected_engine = check_dropdown_click(x, y)
                            if selected_engine:
                                current_search_engine = selected_engine
                                show_dropdown = False
                                last_click_time = current_time
                                print(f"🔄 Выбран поисковик: {current_search_engine}")
                        else:
                            pressed_key = check_key_press(x, y, key_positions)
                            
                            if pressed_key == 'SEARCH_SEL':
                                show_dropdown = not show_dropdown
                                last_click_time = current_time
                                print("📋 Dropdown меню открыто" if show_dropdown else "📋 Dropdown меню закрыто")
                                
                            elif pressed_key:
                                show_dropdown = False
                                kx, ky, kw, kh = key_positions[pressed_key]
                                
                                draw_rounded_rectangle(frame, (kx-2, ky-2), (kx + kw+2, ky + kh+2),
                                            (0, 255, 0), 4, 10)
                                
                                handle_key_press(pressed_key)
    
    pinch_was_active = pinch_active
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == 27:
        break
    elif key == ord('c'):
        text_input = ""
        print("🗑️  Текст очищен")
    
    instruction_text = " Pinch to press"
    text_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.putText(frame, instruction_text,
               ((frame_width - text_size[0]) // 2, frame_height - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('Virtual Keyboard - Gesture Control', frame)

# ============================================================================
# ЗАВЕРШЕНИЕ
# ============================================================================

print("\n👋 Программа завершена. До свидания!")
cap.release()
cv2.destroyAllWindows()
hands.close()