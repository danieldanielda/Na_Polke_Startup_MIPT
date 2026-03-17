from aiogram.filters.command import CommandStart
from aiogram.filters.command import Command
from aiogram import Router
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from src.states import UserState

router = Router()

@router.message(CommandStart())
async def start(message: Message, state: FSMContext):
    await message.answer(
        "📸 Пришлите фото штрихкода косметики или ее название — я постараюсь найти полезную информацию о ней!")
    
    await state.set_state(UserState.waiting_for_photo)
    
@router.message(Command("barcode"))
async def info(message: Message, state: FSMContext):
    await message.answer("📌 Советы:\n\n"
                            "• Убедитесь, что штрихкод чёткий и не размыт\n\n"
                            "• Хорошее освещение — обязательно\n\n"
                            "• Держите камеру параллельно штрихкоду\n\n"
                            "• Избегайте бликов и теней\n\n")
    await state.set_state(UserState.waiting_for_photo)