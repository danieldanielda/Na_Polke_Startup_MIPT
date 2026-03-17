import os
import re
import tempfile
import logging
import asyncio
from aiogram import F, Router
from aiogram.types import Message, ContentType, CallbackQuery
from aiogram.enums import ParseMode
from aiogram.fsm.context import FSMContext

from src.states import UserState
from src.utils import decode_barcode_with_cv2, get_product_analysis_by_barcode, get_product_analysis_by_name, format_analysis_for_telegram
from src.keyboards.keyboard import CustomKeyboard

logger = logging.getLogger(__name__)
router = Router()


def escape_md_v2(text: str) -> str:
    escape_chars = r"_*[]()~`>#+-=|{}.!?"
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)


async def send_analysis_result(message: Message, barcode: str, analysis_result: str | None):
    if not analysis_result:
        await message.reply(
            f"🔍 Штрихкод: <code>{barcode}</code>\n\n"
            "Не удалось получить ответ из базы знаний. Проверьте, что сервис агентов и RAG запущены.",
            parse_mode=ParseMode.HTML
        )
        return

    formatted_analysis = await format_analysis_for_telegram(analysis_result)

    text = f"<b>Штрихкод:</b> <code>{barcode}</code>\n\n{formatted_analysis}"

    keyboard = await CustomKeyboard().translate_to_ru()

    await message.reply(
        text,
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard
    )


@router.message(F.text == "ℹ️ Как снимать штрихкод")
async def how_to_scan(message: Message):
    await message.answer(
        "📌 Советы:\n"
        "• Убедитесь, что штрихкод чёткий и не размыт\n"
        "• Хорошее освещение — обязательно\n"
        "• Держите камеру параллельно штрихкоду\n"
        "• Избегайте бликов и теней"
    )


@router.message(F.text & F.text.regexp(r"^\d+$"))
async def handle_barcode_text(message: Message, state: FSMContext):

    barcode = message.text.strip()

    await state.update_data(
        barcode=barcode,
        product_name=None
    )

    await state.set_state(UserState.waiting_for_analysis_type)

    keyboard = await CustomKeyboard().main_keyboard()

    await message.reply(
        f"Штрихкод: `{barcode}`\n\nВыберите тип анализа:",
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=keyboard
    )


@router.callback_query(F.data.in_(["description", "summary"]))
async def handle_analysis_selection(callback_query: CallbackQuery, state: FSMContext):

    await callback_query.answer()

    data = await state.get_data()
    barcode = data.get("barcode")
    product_name = data.get("product_name")
    analysis_type = callback_query.data

    wait_msg = await callback_query.message.reply(
        "🔍 Ищу продукт и подбираю информацию по базе знаний…"
    )

    try:
        if barcode:
            analysis_result = await get_product_analysis_by_barcode(barcode, analysis_type)
            product_identifier = barcode

        elif product_name:
            analysis_result = await get_product_analysis_by_name(product_name, analysis_type)
            product_identifier = product_name

        else:
            await callback_query.message.reply("❌ Не удалось определить продукт для анализа.")
            return

        await send_analysis_result(
            callback_query.message,
            product_identifier,
            analysis_result
        )

        await state.update_data(
            original_analysis=analysis_result
        )

        keyboard = await CustomKeyboard().after_analysis_keyboard()

        await callback_query.message.answer(
            "Что хотите сделать дальше?",
            reply_markup=keyboard
        )

        await state.set_state(UserState.analysis_finished)

    except Exception as e:
        logger.error(f"Error during product lookup: {e}", exc_info=True)
        await callback_query.message.reply("❌ Произошла ошибка при обработке запроса.")

    finally:
        try:
            await callback_query.bot.delete_message(
                chat_id=callback_query.message.chat.id,
                message_id=wait_msg.message_id
            )
        except Exception:
            pass


@router.callback_query(F.data == "repeat_analysis", UserState.analysis_finished)
async def repeat_analysis(callback: CallbackQuery, state: FSMContext):

    await callback.answer()

    keyboard = await CustomKeyboard().main_keyboard()

    await callback.message.answer(
        "Выберите тип анализа:",
        reply_markup=keyboard
    )

    await state.set_state(UserState.waiting_for_analysis_type)


@router.callback_query(F.data == "new_photo", UserState.analysis_finished)
async def new_photo(callback: CallbackQuery, state: FSMContext):

    await callback.answer()

    await callback.message.answer(
        "📷 Отправьте новое фото со штрихкодом или введите название средства"
    )

    await state.set_state(UserState.waiting_for_photo)


@router.message(F.text, ~F.text.regexp(r"^\d+$"))
async def handle_product_name_text(message: Message, state: FSMContext):

    product_name = message.text.strip()

    await state.update_data(
        product_name=product_name,
        barcode=None
    )

    await state.set_state(UserState.waiting_for_analysis_type)

    keyboard = await CustomKeyboard().main_keyboard()

    await message.reply(
        f"Название продукта: `{product_name}`\n\nВыберите тип анализа:",
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=keyboard
    )


@router.message(UserState.waiting_for_photo, F.content_type == ContentType.PHOTO)
async def handle_photo(message: Message, state: FSMContext):

    photo = message.photo[-1]
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        await message.bot.download(photo, destination=tmp_path)

        barcode = await asyncio.to_thread(decode_barcode_with_cv2, tmp_path)

        if not barcode:
            await message.reply(
                "❌ Не удалось распознать штрихкод.\n\n"
                "Попробуйте сделать фото чётче или в другом ракурсе."
            )
            return

        keyboard = await CustomKeyboard().main_keyboard()

        await message.reply(
            f"Штрихкод: `{barcode}`\n\nВыберите тип анализа:",
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=keyboard
        )

        await state.update_data(barcode=barcode)
        await state.set_state(UserState.waiting_for_analysis_type)

    except Exception as e:
        logger.error(f"Error processing photo: {e}", exc_info=True)
        await message.reply("❌ Ошибка при обработке изображения.")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass