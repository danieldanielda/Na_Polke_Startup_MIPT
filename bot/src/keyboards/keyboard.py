from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils.keyboard import ReplyKeyboardBuilder

class CustomKeyboard:
    
    def __init__(self):
        pass
    
    async def main_keyboard(self) -> InlineKeyboardMarkup:
        balance_keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text='☘️ Безопасность состава', callback_data='description')],
                [InlineKeyboardButton(text='🔍 Полный анализ средства', callback_data='summary')],
            ]
        )
        return balance_keyboard
    
    async def after_analysis_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="🔁 Повторить анализ средства", callback_data="repeat_analysis")],
                [InlineKeyboardButton(text="📷 Загрузить новое фото или название", callback_data="new_photo")],
            ]
        )
        return keyboard
    
    async def translate_to_ru(self):
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="🌍 Перевести на русский",
                        callback_data="translate_ru"
                    )
                ]
            ]
        )

    async def translate_to_en(self):
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="🌍 Translate to English",
                        callback_data="translate_en"
                    )
                ]
            ]
        )