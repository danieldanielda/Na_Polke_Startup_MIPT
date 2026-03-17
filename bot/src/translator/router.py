import logging
from aiogram import F, Router
from aiogram.types import CallbackQuery
from aiogram.fsm.context import FSMContext

from src.states import UserState
from src.services.translate_service import translate_text
from src.utils import format_analysis_for_telegram
from src.keyboards.keyboard import CustomKeyboard

logger = logging.getLogger(__name__)
router = Router()


@router.callback_query(F.data == "translate_ru", UserState.analysis_finished)
async def translate_to_ru(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    data = await state.get_data()
    text = data.get("original_analysis") or data.get("last_analysis")

    if not text:
        await callback.message.answer("❌ Нет текста для перевода.")
        return

    wait_msg = await callback.message.answer("🌍 Перевожу на русский...")
    translated = await translate_text(text, "ru")
    await callback.bot.delete_message(chat_id=callback.message.chat.id, message_id=wait_msg.message_id)

    if not translated:
        await callback.message.answer("❌ Ошибка перевода.")
        return

    formatted_text = await format_analysis_for_telegram(translated)

    keyboard = await CustomKeyboard().translate_to_en()

    await callback.message.edit_text(
        formatted_text,
        reply_markup=keyboard,
        parse_mode="HTML"
    )

    await state.update_data(last_analysis=translated)


@router.callback_query(F.data == "translate_en", UserState.analysis_finished)
async def translate_to_en(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    data = await state.get_data()
    text = data.get("original_analysis") or data.get("last_analysis")

    if not text:
        await callback.message.answer("❌ Нет текста для перевода.")
        return

    wait_msg = await callback.message.answer("🌍 Translating to English...")
    translated = await translate_text(text, "en")
    await callback.bot.delete_message(chat_id=callback.message.chat.id, message_id=wait_msg.message_id)

    if not translated:
        await callback.message.answer("❌ Translation error.")
        return

    formatted_text = await format_analysis_for_telegram(translated)

    keyboard = await CustomKeyboard().translate_to_ru()

    await callback.message.edit_text(
        formatted_text,
        reply_markup=keyboard,
        parse_mode="HTML"
    )

    await state.update_data(last_analysis=translated)