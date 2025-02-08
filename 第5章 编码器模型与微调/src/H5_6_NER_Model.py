from torch import nn
from transformers import BertPreTrainedModel, BertModel

class BertForNER(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.post_init()

    def forward(self, batch_inputs, labels=None):
        bert_output = self.bert(**batch_inputs)
        sequence_output = bert_output.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            attention_mask = batch_inputs.get('attention_mask')
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_func(active_logits, active_labels)
            else:
                loss = loss_func(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits
