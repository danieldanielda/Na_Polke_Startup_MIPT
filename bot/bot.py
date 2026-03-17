import logging
import asyncio
import sys

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand

from src.start_router import router as start_router
from src.parse_barcode.router import router as barcode_router
from src.translator.router import router as translate_router

from config import BotSettings

logger = logging.getLogger(__name__)
settings = BotSettings()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    stream=sys.stdout,
)

bot = Bot(token=settings.tg_api_key)
dp = Dispatcher()

async def setup_bot_commands():
    bot_commands = [
        BotCommand(command="/start", description="Запуск/Перезапуск бота"),
        BotCommand(command="/barcode", description="Как снимать штрихкод")
    ]
    await bot.set_my_commands(bot_commands)
            
async def main():
    
    dp.include_router(start_router)
    dp.include_router(barcode_router)
    dp.include_router(translate_router)
    dp.startup.register(setup_bot_commands)
    
    await dp.start_polling(bot)

if __name__ == "__main__":
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Exit from Bot')