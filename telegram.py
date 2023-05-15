import logging
from aiogram import Bot, Dispatcher, types, executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
import aiosqlite

logging.basicConfig(level=logging.INFO)

API_TOKEN = 'TOKEN'

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())


class Survey(StatesGroup):
    age = State()
    education = State()
    gender = State()
    income = State()


@dp.message_handler(commands='start')
async def cmd_start(message: types.Message):
    await Survey.age.set()


@dp.message_handler(state=Survey.age)
async def process_age(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['age'] = message.text
    await message.answer("How many years of education have you completed?")
    await Survey.next()


@dp.message_handler(state=Survey.education)
async def process_education(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['education'] = message.text
    await message.answer("What is your gender?")
    await Survey.next()


@dp.message_handler(state=Survey.gender)
async def process_gender(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['gender'] = message.text
    await message.answer("What is your income?")
    await Survey.next()


@dp.message_handler(state=Survey.income)
async def process_income(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['income'] = message.text

    async with aiosqlite.connect('survey.db') as db:
        cursor = await db.cursor()
        await cursor.execute(
            "CREATE TABLE IF NOT EXISTS survey (age integer, education integer, gender text, income integer)")
        await cursor.execute("INSERT INTO survey (age, education, gender, income) VALUES (?, ?, ?, ?)",
                             (data['age'], data['education'], data['gender'], data['income']))
        await db.commit()

    await message.answer("Thank you for your responses!")
    await state.finish()


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
