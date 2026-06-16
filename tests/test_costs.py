# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway._costs import Cost, CostSession


def test_cost():
    cost_1 = Cost(time=10, memory=20)
    cost_2 = Cost(time=30, memory=40)

    with CostSession()() as session_1:
        total_1 = session_1.sum()
        cost_1.commit()
        total_2 = session_1.sum()

        assert total_2 - total_1 == cost_1

        with CostSession()() as session_2:

            total_3 = session_2.sum()
            assert total_3 == Cost.zero()

            cost_2.commit()
            total_4 = session_2.sum()
            assert total_4 - total_3 == cost_2

            assert session_1.sum() == total_2 + total_4

        assert session_1.sum() == total_2


def test_cost_session():
    sess_1 = CostSession()
    with sess_1():
        assert CostSession.current() is sess_1

        with CostSession()() as sess_2:
            assert CostSession.current() is sess_2

        assert CostSession.current() is sess_1
