import time
from collections import defaultdict

from loguru import logger

from flcore.clients.clientdistill import clientDistill
from flcore.servers.serverbase import Server


class FedDistill(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientDistill)

        logger.info(f"Join ratio / total clients: {self.join_ratio} / {self.num_clients}")
        logger.info("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes
        self.global_logits = [None for _ in range(args.num_classes)]


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i%self.eval_gap == 0:
                logger.info(f"-------------Round number: {i}-------------")
                logger.info("Evaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_logits()
            self.global_logits = logit_aggregation(self.uploaded_logits)
            self.send_logits()

            self.Budget.append(time.time() - s_t)
            logger.info('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        logger.info("Best accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        logger.info(max(self.rs_test_acc))
        logger.info(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        

    def send_logits(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_logits(self.global_logits)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_logits(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_logits = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_logits.append(client.logits)


# https://github.com/yuetan031/fedlogit/blob/main/lib/utils.py#L221
def logit_aggregation(local_logits_list):
    agg_logits_label = defaultdict(list)
    for local_logits in local_logits_list:
        for label in local_logits.keys():
            agg_logits_label[label].append(local_logits[label])

    for [label, logit_list] in agg_logits_label.items():
        if len(logit_list) > 1:
            logit = 0 * logit_list[0].data
            for i in logit_list:
                logit += i.data
            agg_logits_label[label] = logit / len(logit_list)
        else:
            agg_logits_label[label] = logit_list[0].data

    return agg_logits_label